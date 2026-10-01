"""Pre-publication impact review of a ruleset revision (draft or any stored one).

Before activating a *target* revision, a caller submits the ruleset name
(in the URL), the revision the caller created the draft against
(``base_revision`` -- the active revision it read at draft time, ``null``
before the ruleset's first publication) and 1..50 persisted assessment
ids.  The store opens a single SQLite read transaction, re-reads the
active revision inside it and accepts the batch only when every
assessment

1. belongs to that ruleset, and
2. recorded that exact active revision.

Any stale baseline, unknown target revision, unknown assessment,
foreign-ruleset assessment or assessment recorded against a different
revision fails the *whole* request with HTTP 409 -- no partial report is
ever produced.

For the accepted batch each assessment is re-evaluated against the
target revision's immutable rules snapshot.  The report gives, per
assessment sorted by id, the old/new status and ``summary_hash``, the
adopted branches (before/after) and the obligation differences, plus a
summary of the items that would move from a release (``status == "ok"``)
to a conflict or another non-releasable/error status.

Everything in the report is produced from sorted structures, so the
request ordering never changes the result or its ``report_hash``.
"""

from .common import ClientError, stable_hash
from .solver import assess_normalized

# A review covers at most this many assessments per call, keeping the
# re-evaluation work bounded; the bound is enforced as a 400 envelope
# error (it holds regardless of database state).
MAX_REVIEW_ASSESSMENTS = 50


def validate_review_body(body):
    """Validate the review request envelope.

    Returns the ``(base_revision, assessment_ids)`` pair; the ids must be
    listed at most once each.  Ruleset-level consistency (the base
    revision still being active, every id belonging to the ruleset and
    recording that revision) is verified inside the store's read
    transaction, not here.
    """
    if not isinstance(body, dict):
        raise ClientError("request body must be a JSON object")
    if "base_revision" not in body:
        raise ClientError(
            "impact review requires 'base_revision'"
            " (the active revision read when the draft was created,"
            " null before first publication)"
        )
    base_revision = body["base_revision"]
    if base_revision is not None and not (
        isinstance(base_revision, str) and base_revision
    ):
        raise ClientError("base_revision must be a non-empty string or null")
    ids = body.get("assessment_ids")
    if not isinstance(ids, list) or not ids:
        raise ClientError(
            "assessment_ids must be a non-empty array of 1..%d assessment ids"
            % MAX_REVIEW_ASSESSMENTS
        )
    if len(ids) > MAX_REVIEW_ASSESSMENTS:
        raise ClientError(
            "impact review accepts at most %d assessment ids (got %d)"
            % (MAX_REVIEW_ASSESSMENTS, len(ids))
        )
    seen = set()
    for i, aid in enumerate(ids):
        if not isinstance(aid, str) or not aid:
            raise ClientError("assessment_ids[%d] must be a non-empty string" % i)
        if aid in seen:
            raise ClientError("assessment_ids contains duplicate id %r" % aid)
        seen.add(aid)
    return base_revision, ids


# ---------------------------------------------------------------------------
# result projections
# ---------------------------------------------------------------------------

def _adopted_of(result):
    """Adopted branches of a result; null for any non-releasable status.

    An ``ok`` result carries an ``adopted`` mapping directly; every other
    status has no release conclusion, so no adopted branches exist.
    """
    if result.get("status") == "ok":
        return dict(result.get("adopted", {}))
    return None


def _component_obligations(result):
    """Component id -> sorted obligation list as recorded by *result*.

    Both ``ok`` results (``components``) and conflict explanations
    (``confirmed_subgraph.components``) carry per-component obligations;
    components absent from either contribute no obligations.
    """
    out = {}
    if result.get("status") == "ok":
        components = result.get("components", [])
    else:
        components = result.get("confirmed_subgraph", {}).get("components", [])
    for c in components or []:
        out[c.get("id")] = sorted(c.get("obligations", []))
    return out


def _obligation_diff(before, after):
    """Added/removed artifact and per-component obligations."""
    old_art = set(before.get("artifact_obligations", []))
    new_art = set(after.get("artifact_obligations", []))
    old_comp = _component_obligations(before)
    new_comp = _component_obligations(after)
    components = []
    for cid in sorted(set(old_comp) | set(new_comp)):
        added = sorted(set(new_comp.get(cid, [])) - set(old_comp.get(cid, [])))
        removed = sorted(set(old_comp.get(cid, [])) - set(new_comp.get(cid, [])))
        if added or removed:
            components.append(
                {"component": cid, "added": added, "removed": removed}
            )
    return {
        "artifact_obligations": {
            "added": sorted(new_art - old_art),
            "removed": sorted(old_art - new_art),
        },
        "components": components,
    }


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------

def build_impact_report(name, target_revision, base_revision, rows, target_rules):
    """Re-evaluate every accepted assessment against the target snapshot.

    *rows* are the persisted assessment rows fetched inside one read
    transaction (``id`` / ``request`` / ``result`` plus provenance); they
    are re-sorted here so the request's id ordering cannot influence the
    report.  Re-evaluation happens outside the database transaction -- it
    is a pure function of each pinned normalized request and the target
    revision's immutable snapshot.
    """
    items = []
    transitions = {}
    regressed_to_conflict = []
    regressed_to_error = []

    for row in sorted(rows, key=lambda r: r["id"]):
        aid = row["id"]
        before = row["result"]
        after = assess_normalized(row["request"], target_rules)

        old_status = before.get("status")
        new_status = after.get("status")
        # A regression is a release conclusion that disappears: the old
        # result was releasable and the target-snapshot result is not.
        regressed = bool(old_status == "ok" and new_status != "ok")
        if regressed:
            if new_status == "conflict":
                regressed_to_conflict.append(aid)
            else:
                regressed_to_error.append(aid)
        transitions[(old_status, new_status)] = transitions.get(
            (old_status, new_status), 0
        ) + 1

        items.append(
            {
                "assessment_id": aid,
                "old_status": old_status,
                "new_status": new_status,
                "old_summary_hash": before.get("summary_hash"),
                "new_summary_hash": after.get("summary_hash"),
                "old_adopted": _adopted_of(before),
                "new_adopted": _adopted_of(after),
                "obligation_diff": _obligation_diff(before, after),
                "regressed": regressed,
            }
        )

    # "release -> conflict or error" is the headline: every item that was
    # releasable under the active revision but no longer is, split into the
    # conflict bucket and the other non-releasable evaluation-error bucket.
    regressed_to_conflict.sort()
    regressed_to_error.sort()
    regressed_all = regressed_to_conflict + regressed_to_error

    status_changes = [
        {"from": old, "to": new, "count": count}
        for (old, new), count in sorted(transitions.items())
        if old != new
    ]
    report = {
        "ruleset": name,
        "target_revision": target_revision,
        "base_revision": base_revision,
        "assessed_count": len(items),
        "items": items,
        "summary": {
            "release_to_conflict": regressed_to_conflict,
            "release_to_error": regressed_to_error,
            "release_to_conflict_or_error": regressed_all,
            "release_to_conflict_or_error_count": len(regressed_all),
            "status_changes": status_changes,
            "status_changed_count": sum(
                c["count"] for c in status_changes
            ),
        },
    }
    report["report_hash"] = stable_hash(report)
    return report
