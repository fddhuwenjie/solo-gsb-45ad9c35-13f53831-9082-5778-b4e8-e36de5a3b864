"""Immutable license release batch registration.

A release batch names 1..50 already-persisted, ``status == "ok"``
assessments and, for each one, lists every artifact (component) it
releases with its declared version and dependency relationships.  The
service verifies the declarations against each assessment's pinned
snapshot inside **one SQLite transaction**, locks the rule revisions the
assessments were recorded against, and then merges the artifacts:

* an artifact id may appear only once per assessment and may not name two
  different components inside one batch (``duplicate-artifact`` -> 422);
* the *same* component (artifact id) shared across assessments is
  deduplicated to one manifest entry; its name, version and adopted
  license must agree in every assessment it appears in, otherwise the
  whole batch is refused (``component-mismatch`` -> 422);
* artifact obligations are the union of the per-component obligations
  recorded by each contributing assessment, and dependency relationships
  are the union of the verified per-assessment edges.

The resulting manifest -- sorted artifacts and sorted contributing
assessments, everything derived from sorted structures -- is frozen under
a ``batch_hash`` (SHA-256 over the canonical manifest).  Re-registering
the same ``batch_id`` is idempotent only when the rebuilt content is
byte-for-byte identical (equal ``batch_hash``); a reused id with different
content fails with 409.  Because the manifest and every rule snapshot it
was built from are persisted in the registration transaction, later rule
publications can never change a registered batch.
"""

import re

from .common import ClientError, UnprocessableError, stable_hash

# A batch covers at most this many assessments per call; the bound is an
# envelope-level 400 error (it holds regardless of database state).
MAX_BATCH_ASSESSMENTS = 50

# Batch ids are caller-chosen unique ids (UUIDs, release tickets, ...);
# keep the alphabet URL/path-safe and the length bounded.
_BATCH_ID_RE = re.compile(r"^[A-Za-z0-9._:/~-]{1,128}$")


# ---------------------------------------------------------------------------
# envelope validation
# ---------------------------------------------------------------------------

def validate_batch_body(body):
    """Validate the release-batch request envelope.

    Returns ``(batch_id, entries)`` where each entry is
    ``{"assessment_id", "artifacts": [{"id", "version", "dependencies":
    [...]}]}`` with dependency targets deduplicated and sorted; artifact
    order inside an entry is normalized to sorted id order downstream.

    Structural problems (non-object body, missing/mistyped fields, empty
    or oversized lists, duplicate assessment ids, malformed batch id)
    raise :class:`ClientError` (400).  Content conflicts that only become
    meaningful against persisted state -- duplicate artifact ids within
    one assessment entry -- raise :class:`UnprocessableError` (422);
    unknown/missing/mismatching artifacts and cross-assessment component
    inconsistency are checked against the snapshots later.
    """
    if not isinstance(body, dict):
        raise ClientError("request body must be a JSON object")
    batch_id = body.get("batch_id")
    if not isinstance(batch_id, str) or not _BATCH_ID_RE.match(batch_id or ""):
        raise ClientError(
            "batch_id must be a non-empty string of 1..128 URL-safe characters"
            " (letters, digits and . _ : / ~ -)"
        )
    raw_entries = body.get("assessments")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ClientError(
            "assessments must be a non-empty array of 1..%d assessment entries"
            % MAX_BATCH_ASSESSMENTS
        )
    if len(raw_entries) > MAX_BATCH_ASSESSMENTS:
        raise ClientError(
            "a release batch accepts at most %d assessments (got %d)"
            % (MAX_BATCH_ASSESSMENTS, len(raw_entries))
        )

    entries = []
    seen_assessments = set()
    for i, raw_entry in enumerate(raw_entries):
        where = "assessments[%d]" % i
        if not isinstance(raw_entry, dict):
            raise ClientError("%s must be an object" % where)
        aid = raw_entry.get("assessment_id")
        if not isinstance(aid, str) or not aid:
            raise ClientError("%s.assessment_id must be a non-empty string" % where)
        if aid in seen_assessments:
            raise ClientError(
                "%s lists assessment %r more than once" % (where, aid)
            )
        seen_assessments.add(aid)

        raw_artifacts = raw_entry.get("artifacts")
        if not isinstance(raw_artifacts, list) or not raw_artifacts:
            raise ClientError("%s.artifacts must be a non-empty array" % where)

        artifacts = []
        seen_artifacts = set()
        for j, raw_art in enumerate(raw_artifacts):
            awhere = "%s.artifacts[%d]" % (where, j)
            if not isinstance(raw_art, dict):
                raise ClientError("%s must be an object" % awhere)
            art_id = raw_art.get("id")
            if not isinstance(art_id, str) or not art_id.strip():
                raise ClientError("%s.id must be a non-empty string" % awhere)
            art_id = art_id.strip()
            version = raw_art.get("version", "")
            if not isinstance(version, str):
                raise ClientError("%s.version must be a string" % awhere)
            raw_deps = raw_art.get("dependencies", [])
            if not isinstance(raw_deps, list):
                raise ClientError("%s.dependencies must be an array" % awhere)
            deps = []
            for k, dep in enumerate(raw_deps):
                if not isinstance(dep, str) or not dep.strip():
                    raise ClientError(
                        "%s.dependencies[%d] must be a non-empty artifact id"
                        % (awhere, k)
                    )
                deps.append(dep.strip())
            if art_id in seen_artifacts:
                raise UnprocessableError(
                    "assessment %r lists artifact %r more than once" % (aid, art_id),
                    {
                        "reason": "duplicate-artifact",
                        "assessment_id": aid,
                        "artifact_id": art_id,
                    },
                )
            seen_artifacts.add(art_id)
            artifacts.append(
                {
                    "id": art_id,
                    "version": version,
                    "dependencies": sorted(set(deps)),
                }
            )
        artifacts.sort(key=lambda a: a["id"])
        entries.append({"assessment_id": aid, "artifacts": artifacts})
    return batch_id, entries


# ---------------------------------------------------------------------------
# per-assessment declaration verification
# ---------------------------------------------------------------------------

def verify_entry(entry, request, result):
    """Verify one assessment's artifact declarations against its snapshot.

    Every declared artifact must name a component of the assessment, the
    declaration set must cover *every* component (no silent dropping),
    the declared version must equal the component's recorded version and
    the declared dependency targets must equal the component's outgoing
    dependency edges in the pinned request.  Any discrepancy raises
    :class:`UnprocessableError` (422) carrying a structured reason; the
    whole batch is refused and nothing is written.
    """
    aid = entry["assessment_id"]
    info = {c["id"]: c for c in request["components"]}
    edges = {(d["from"], d["to"]) for d in request["dependencies"]}
    declared = {a["id"]: a for a in entry["artifacts"]}

    unknown = sorted(a for a in declared if a not in info)
    if unknown:
        raise UnprocessableError(
            "assessment %r declares artifacts absent from the assessment: %s"
            % (aid, ", ".join(unknown)),
            {"reason": "artifact-unknown", "assessment_id": aid, "artifact_ids": unknown},
        )
    missing = sorted(cid for cid in info if cid not in declared)
    if missing:
        raise UnprocessableError(
            "assessment %r omits artifacts that must be registered: %s"
            % (aid, ", ".join(missing)),
            {"reason": "artifact-incomplete", "assessment_id": aid, "artifact_ids": missing},
        )

    version_conflicts = []
    dep_conflicts = []
    for art_id in sorted(declared):
        art = declared[art_id]
        actual_version = info[art_id]["version"]
        if art["version"] != actual_version:
            version_conflicts.append(
                {
                    "artifact_id": art_id,
                    "declared_version": art["version"],
                    "recorded_version": actual_version,
                }
            )
        expected = sorted(t for f, t in edges if f == art_id)
        if list(art["dependencies"]) != expected:
            dep_conflicts.append(
                {
                    "artifact_id": art_id,
                    "declared_dependencies": list(art["dependencies"]),
                    "recorded_dependencies": expected,
                }
            )
    if version_conflicts:
        raise UnprocessableError(
            "assessment %r declares versions that disagree with its snapshot" % aid,
            {
                "reason": "artifact-version-mismatch",
                "assessment_id": aid,
                "conflicts": version_conflicts,
            },
        )
    if dep_conflicts:
        raise UnprocessableError(
            "assessment %r declares dependencies that disagree with its snapshot"
            % aid,
            {
                "reason": "artifact-dependency-mismatch",
                "assessment_id": aid,
                "conflicts": dep_conflicts,
            },
        )


# ---------------------------------------------------------------------------
# deterministic batch merge
# ---------------------------------------------------------------------------

def build_batch_manifest(batch_id, items):
    """Merge verified assessment items into one immutable batch manifest.

    *items* pairs each validated entry with its persisted assessment row
    (``request`` / ``result`` / ``ruleset`` / ``revision``).  Shared
    artifacts (same component id in several assessments) collapse onto one
    entry whose obligations and dependency targets are the union of every
    occurrence; a shared component whose name, version or adopted license
    disagrees between two assessments fails the whole merge with
    ``component-mismatch`` (422).
    """
    merged = {}  # artifact id -> merged record
    first_seen = {}  # artifact id -> first assessment id
    order = []  # first-appearance order (manifest is sorted later)

    mismatches = []
    duplicate_components = []

    for item in sorted(items, key=lambda it: it["entry"]["assessment_id"]):
        entry = item["entry"]
        aid = entry["assessment_id"]
        request = item["request"]
        result = item["result"]
        info = {c["id"]: c for c in request["components"]}
        obligations_by_id = {
            c["id"]: list(c.get("obligations", []))
            for c in result.get("components", [])
        }
        adopted = result.get("adopted", {})
        edges = {(d["from"], d["to"]) for d in request["dependencies"]}

        component_ids = {a["id"] for a in entry["artifacts"]}
        for art_id in sorted(component_ids):
            if art_id in merged and first_seen.get(art_id) == aid:
                # Two declarations of one component inside a single
                # assessment -- envelope validation rejects exact dup ids,
                # so this is a defensive guard, not a normal path.
                duplicate_components.append(
                    {"assessment_id": aid, "artifact_id": art_id}
                )
                continue
            component = {
                "name": info[art_id]["name"],
                "version": info[art_id]["version"],
                "license": adopted.get(art_id, ""),
            }
            if art_id not in merged:
                first_seen[art_id] = aid
                order.append(art_id)
                merged[art_id] = {
                    "id": art_id,
                    "name": component["name"],
                    "version": component["version"],
                    "license": component["license"],
                    "obligations": set(obligations_by_id.get(art_id, [])),
                    "dependencies": {t for f, t in edges if f == art_id},
                    "assessments": {aid},
                }
                continue
            existing = merged[art_id]
            for field in ("name", "version", "license"):
                if existing[field] != component[field]:
                    mismatches.append(
                        {
                            "artifact_id": art_id,
                            "field": field,
                            "first_assessment_id": first_seen[art_id],
                            "existing_value": existing[field],
                            "assessment_id": aid,
                            "conflicting_value": component[field],
                        }
                    )
            existing["obligations"].update(obligations_by_id.get(art_id, []))
            existing["dependencies"].update(t for f, t in edges if f == art_id)
            existing["assessments"].add(aid)

    if duplicate_components:
        raise UnprocessableError(
            "the batch lists the same artifact more than once in one assessment",
            {"reason": "duplicate-artifact", "conflicts": duplicate_components},
        )
    if mismatches:
        conflicts = sorted(
            mismatches,
            key=lambda m: (m["artifact_id"], m["field"], m["assessment_id"]),
        )
        raise UnprocessableError(
            "shared components disagree across assessments: %s"
            % ", ".join(
                "%s(%s)" % (m["artifact_id"], m["field"]) for m in conflicts
            ),
            {"reason": "component-mismatch", "conflicts": conflicts},
        )

    artifacts = [
        {
            "id": art_id,
            "name": merged[art_id]["name"],
            "version": merged[art_id]["version"],
            "license": merged[art_id]["license"],
            "obligations": sorted(merged[art_id]["obligations"]),
            "dependencies": sorted(merged[art_id]["dependencies"]),
            "assessments": sorted(merged[art_id]["assessments"]),
        }
        for art_id in sorted(merged)
    ]
    assessments = [
        {
            "assessment_id": item["entry"]["assessment_id"],
            "ruleset": item["ruleset"],
            "ruleset_revision": item["revision"],
            "summary_hash": item["result"].get("summary_hash"),
            "artifacts": sorted(a["id"] for a in item["entry"]["artifacts"]),
        }
        for item in sorted(items, key=lambda it: it["entry"]["assessment_id"])
    ]
    content = {
        "artifact_count": len(artifacts),
        "assessment_count": len(assessments),
        "artifacts": artifacts,
        "assessments": assessments,
    }
    # The batch_hash covers the content only -- never the caller-chosen
    # batch_id -- so equal content (regardless of its id or the request's
    # ordering) always yields the same hash, and an idempotent retry can
    # prove equality on batch_hash alone.
    manifest = dict(content)
    manifest["batch_id"] = batch_id
    manifest["batch_hash"] = stable_hash(content)
    return manifest
