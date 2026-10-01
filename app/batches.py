"""Immutable license release batches: envelope validation and manifest merge.

A release batch registers 1..50 *releasable* (``status == "ok"``)
assessments as one publication.  Each request entry names one artifact
(the component id within that assessment that is being published), its
version and the dependencies the publication declares for it.

The store reads and locks every referenced assessment and the immutable
ruleset revision each one was recorded against **inside one SQLite write
transaction**, then calls :func:`assemble_manifest` there:

* an assessment that does not exist or is not releasable fails the whole
  batch (409);
* a snapshot that cannot be verified (inline-rule or legacy rows without
  a pinned, revision-backed rules snapshot) fails the whole batch (422);
* the named artifact must exist in the assessment and its published
  version/license and every declared dependency must match the pinned
  snapshot -- otherwise the batch is unprocessable (422);
* the same artifact may not be published twice in one batch, and a shared
  component (same component id in different assessments) must carry the
  same name, version and adopted license across all of them -- otherwise
  the merge conflicts (409);
* shared components are deduplicated: artifacts and the union of their
  component obligations are merged into one manifest.

Everything in the manifest is produced from sorted structures, so the
request entry ordering never changes the manifest or its ``batch_hash``.
"""

import re

from .common import ClientError, ConflictError, stable_hash
from .fulfillment import UnprocessableError

MAX_BATCH_ENTRIES = 50
_BATCH_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def validate_batch_body(body):
    """Validate the release-batch request envelope.

    Returns ``(batch_id, entries)`` where each entry is
    ``{"assessment_id", "id", "version", "dependencies"}`` with
    deduplicated, sorted dependencies.  Only request-shape rules live
    here (they hold regardless of database state -> 400); every stateful
    and snapshot rule is enforced inside the store transaction.
    """
    if not isinstance(body, dict):
        raise ClientError("request body must be a JSON object")
    batch_id = body.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise ClientError("batch_id must be a non-empty string")
    batch_id = batch_id.strip()
    if not _BATCH_ID_RE.match(batch_id):
        raise ClientError(
            "batch_id may contain only letters, digits, dot, underscore"
            " and hyphen (1..128 characters)"
        )

    entries_raw = body.get("entries")
    if not isinstance(entries_raw, list) or not entries_raw:
        raise ClientError(
            "entries must be a non-empty array of 1..%d release entries"
            % MAX_BATCH_ENTRIES
        )
    if len(entries_raw) > MAX_BATCH_ENTRIES:
        raise ClientError(
            "a release batch accepts at most %d entries (got %d)"
            % (MAX_BATCH_ENTRIES, len(entries_raw))
        )

    entries = []
    seen_artifacts = set()
    for i, raw in enumerate(entries_raw):
        if not isinstance(raw, dict):
            raise ClientError("entries[%d] must be an object" % i)
        aid = raw.get("assessment_id")
        if not isinstance(aid, str) or not aid:
            raise ClientError(
                "entries[%d].assessment_id must be a non-empty string" % i
            )
        artifact_id = raw.get("id")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ClientError("entries[%d].id must be a non-empty string" % i)
        artifact_id = artifact_id.strip()
        version = raw.get("version")
        if not isinstance(version, str) or not version.strip():
            raise ClientError(
                "entries[%d].version must be a non-empty string" % i
            )
        version = version.strip()
        deps_raw = raw.get("dependencies", [])
        if not isinstance(deps_raw, list):
            raise ClientError("entries[%d].dependencies must be an array" % i)
        deps = []
        for j, dep in enumerate(deps_raw):
            if not isinstance(dep, str) or not dep.strip():
                raise ClientError(
                    "entries[%d].dependencies[%d] must be a non-empty string" % (i, j)
                )
            deps.append(dep.strip())
        if len(set(deps)) != len(deps):
            raise ClientError(
                "entries[%d].dependencies contains a duplicate component id" % i
            )
        # One artifact id may appear only once in a submitted batch; the
        # cross-assessment meaning of that is a stateful merge conflict,
        # while a duplicated request entry never has a canonical form.
        if artifact_id in seen_artifacts:
            raise ClientError(
                "entries contains duplicate artifact id %r" % artifact_id
            )
        seen_artifacts.add(artifact_id)
        entries.append(
            {
                "assessment_id": aid,
                "id": artifact_id,
                "version": version,
                "dependencies": sorted(deps),
            }
        )
    return batch_id, entries


def canonical_entries(entries):
    """Order-independent fingerprint of the submitted batch content.

    Entries are sorted (and their dependencies already sorted), so two
    submissions differing only in entry order carry the same fingerprint.
    """
    return sorted(
        (
            {
                "assessment_id": e["assessment_id"],
                "id": e["id"],
                "version": e["version"],
                "dependencies": list(e["dependencies"]),
            }
            for e in entries
        ),
        key=lambda e: (e["id"], e["assessment_id"]),
    )


def assemble_manifest(batch_id, loaded):
    """Merge the transaction-locked assessment rows into a batch manifest.

    *loaded* is a list aligned with the validated entries of
    ``(entry, request, result, snapshot)`` tuples; the caller has already
    proven each assessment exists, is ``status == "ok"`` and carries a
    verifiable, revision-backed rules snapshot (``snapshot`` is its
    ``{"ruleset", "revision", "rules_hash"}`` provenance).  Pure snapshot
    checks (artifact/version/license/dependency mismatch -> 422) and
    cross-entry merge checks (duplicate artifact, shared-component
    disagreement -> 409) happen here.

    The result is order-independent: callers may submit entries in any
    order and receive byte-identical manifests and ``batch_hash`` values.
    """
    artifacts = []
    # component id -> merged component record
    components = {}
    snapshots = {}

    for entry, request, result, snapshot in loaded:
        aid = entry["assessment_id"]
        snapshots[aid] = dict(snapshot)
        artifact_id = entry["id"]
        info = {c["id"]: c for c in request["components"]}
        comp_result = {c["id"]: c for c in result.get("components", [])}
        adopted = result.get("adopted", {})

        if artifact_id not in info:
            raise UnprocessableError(
                "entry for assessment %r publishes artifact %r, but that "
                "component is not part of the assessment snapshot"
                % (aid, artifact_id)
            )
        artifact_meta = info[artifact_id]
        if entry["version"] != artifact_meta["version"]:
            raise UnprocessableError(
                "artifact %r version %r does not match the version recorded "
                "in assessment %r (%r)"
                % (artifact_id, entry["version"], aid, artifact_meta["version"])
            )
        license_expr = adopted.get(artifact_id)
        if not isinstance(license_expr, str) or not license_expr:
            # Defensive: an ok result always adopts every component; a row
            # without an adopted license cannot be released from.
            raise UnprocessableError(
                "assessment %r provides no adopted license for artifact %r"
                % (aid, artifact_id)
            )
        unknown_deps = sorted(d for d in entry["dependencies"] if d not in info)
        if unknown_deps:
            raise UnprocessableError(
                "artifact %r (assessment %r) declares dependencies absent from "
                "the assessment snapshot: %s"
                % (artifact_id, aid, ", ".join(unknown_deps))
            )
        if artifact_id in entry["dependencies"]:
            raise UnprocessableError(
                "artifact %r (assessment %r) cannot depend on itself"
                % (artifact_id, aid)
            )

        artifacts.append(
            {
                "id": artifact_id,
                "name": artifact_meta["name"],
                "version": artifact_meta["version"],
                "license": license_expr,
                "dependencies": list(entry["dependencies"]),
                "obligations": sorted(result.get("artifact_obligations", [])),
                "assessment_id": aid,
            }
        )

        # Every component of a contributing assessment joins the merged
        # component set -- including declared dependencies that are not
        # themselves published as artifacts -- so shared-component
        # obligations are carried into the batch.
        for cid, meta in info.items():
            obligations = sorted(comp_result.get(cid, {}).get("obligations", []))
            existing = components.get(cid)
            if existing is None:
                components[cid] = {
                    "id": cid,
                    "name": meta["name"],
                    "version": meta["version"],
                    "license": adopted.get(cid, ""),
                    "obligations": set(obligations),
                    "assessments": [aid],
                }
                continue
            for field, label in (
                ("name", "name"),
                ("version", "version"),
                ("license", "license"),
            ):
                new_value = meta[field] if field != "license" else adopted.get(cid, "")
                if existing[field] != new_value:
                    raise ConflictError(
                        "shared component %r has an inconsistent %s across "
                        "assessments: %r vs %r"
                        % (cid, label, existing[field], new_value),
                        {
                            "reason": "component-conflict",
                            "component": cid,
                            "field": field,
                            "assessments": sorted(
                                set(existing["assessments"]) | {aid}
                            ),
                        },
                    )
            existing["obligations"].update(obligations)
            if aid not in existing["assessments"]:
                existing["assessments"].append(aid)

    # Duplicate artifact: the same artifact id published by two different
    # entries (the within-request duplicate is rejected at the envelope).
    seen = {}
    for art in artifacts:
        if art["id"] in seen:
            raise ConflictError(
                "artifact %r is published more than once in the batch "
                "(assessments %s and %s)"
                % (art["id"], seen[art["id"]], art["assessment_id"]),
                {
                    "reason": "duplicate-artifact",
                    "artifact": art["id"],
                    "assessment_ids": sorted(
                        {seen[art["id"]], art["assessment_id"]}
                    ),
                },
            )
        seen[art["id"]] = art["assessment_id"]

    merged_artifacts = sorted(artifacts, key=lambda a: a["id"])
    # The batch-level artifact obligations are the union of every
    # contributing assessment's release-level obligations: publishing the
    # batch means shipping every artifact together with the dependencies
    # its snapshot pins.
    batch_artifact_obligations = set()
    for art in merged_artifacts:
        batch_artifact_obligations.update(art["obligations"])
    merged_components = []
    for cid in sorted(components):
        c = components[cid]
        merged_components.append(
            {
                "id": cid,
                "name": c["name"],
                "version": c["version"],
                "license": c["license"],
                "obligations": sorted(c["obligations"]),
                "assessments": sorted(c["assessments"]),
            }
        )

    manifest = {
        "batch_id": batch_id,
        "artifacts": merged_artifacts,
        "artifact_obligations": sorted(batch_artifact_obligations),
        "components": merged_components,
        # The immutable ruleset revision each assessment was recorded
        # against, read and locked in the registering transaction; a later
        # rules publication appends revisions but cannot change these.
        "rule_revisions": [snapshots[aid] for aid in sorted(snapshots)],
    }
    manifest["batch_hash"] = stable_hash(manifest)
    return manifest
