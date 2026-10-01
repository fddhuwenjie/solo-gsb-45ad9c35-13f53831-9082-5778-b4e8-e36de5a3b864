"""SQLite persistence layer (stdlib sqlite3, WAL mode, lock-guarded writes)."""

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone

from .common import ConflictError, NotFound, stable_hash
from .revisions import revision_id
from .solver import assess_normalized

# user_version history:
#   0 -- pre-revision databases (rulesets table + optional rules snapshot)
#   2 -- immutable revisions: ruleset_revisions / ruleset_active /
#        ruleset_activations and assessments.revision provenance
#   3 -- immutable release batches: release_batches /
#        release_batch_assessments, frozen against later rule publications
SCHEMA_VERSION = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS rulesets (
    name       TEXT PRIMARY KEY,
    body       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ruleset_revisions (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    ruleset_name TEXT NOT NULL,
    revision     TEXT NOT NULL,
    body         TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    base_revision TEXT,
    created_at   TEXT NOT NULL,
    UNIQUE (ruleset_name, revision)
);
CREATE TABLE IF NOT EXISTS ruleset_active (
    ruleset_name TEXT PRIMARY KEY,
    revision     TEXT NOT NULL,
    activated_at TEXT NOT NULL,
    FOREIGN KEY (ruleset_name, revision)
        REFERENCES ruleset_revisions(ruleset_name, revision)
);
CREATE TABLE IF NOT EXISTS ruleset_activations (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    ruleset_name TEXT NOT NULL,
    revision     TEXT NOT NULL,
    previous_revision TEXT,
    activated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assessments (
    id           TEXT PRIMARY KEY,
    ruleset      TEXT,
    rules        TEXT,
    revision     TEXT,
    request      TEXT NOT NULL,
    result       TEXT NOT NULL,
    status       TEXT NOT NULL,
    summary_hash TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS confirmations (
    assessment_id TEXT PRIMARY KEY,
    body          TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS release_batches (
    batch_id     TEXT PRIMARY KEY,
    manifest     TEXT NOT NULL,
    batch_hash   TEXT NOT NULL,
    created      INTEGER NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS release_batch_assessments (
    batch_id      TEXT NOT NULL,
    assessment_id TEXT NOT NULL,
    ruleset       TEXT,
    revision      TEXT,
    rules         TEXT NOT NULL,
    seq           INTEGER NOT NULL,
    PRIMARY KEY (batch_id, assessment_id),
    FOREIGN KEY (batch_id) REFERENCES release_batches(batch_id)
);
CREATE INDEX IF NOT EXISTS idx_release_batch_assessments_assessment
    ON release_batch_assessments(assessment_id);
"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    """Thread-safe store; a fresh connection is used per operation."""

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(SCHEMA)
            self._migrate(conn)
            conn.commit()
        finally:
            conn.close()

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # -- schema migration -------------------------------------------------

    def _migrate(self, conn):
        """Upgrade older databases across three migration steps.

        Step 1 (any database): pin rules snapshots onto legacy assessment
        rows whose original rules can be *proven* identical (see
        :meth:`_verified_legacy_rules`).

        Step 2 (user_version < 2): introduce immutable revisions.  Every
        legacy ``rulesets`` row becomes the first active revision of that
        ruleset (the revision id is the content hash, so future identical
        PUTs/drafts collapse onto it), and each assessment is stamped with
        the revision of its pinned snapshot whenever it can be derived.
        Neither move loses or rewrites historical content.

        Step 3 (user_version < 3): create the release-batch tables.  No
        historical content is transformed -- batches only ever *append*.
        """
        self._migrate_assessment_snapshots(conn)
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version < 2:
            self._migrate_revisions(conn)
        if version < SCHEMA_VERSION:
            conn.execute("PRAGMA user_version=%d" % SCHEMA_VERSION)

    def _migrate_assessment_snapshots(self, conn):
        cols = [r[1] for r in conn.execute("PRAGMA table_info(assessments)")]
        if "rules" not in cols:
            conn.execute("ALTER TABLE assessments ADD COLUMN rules TEXT")
        rows = conn.execute(
            "SELECT id, ruleset, request, result FROM assessments WHERE rules IS NULL"
        ).fetchall()
        for row in rows:
            rules = self._verified_legacy_rules(conn, row)
            if rules is not None:
                conn.execute(
                    "UPDATE assessments SET rules = ? WHERE id = ?",
                    (json.dumps(rules, sort_keys=True), row["id"]),
                )

    def _migrate_revisions(self, conn):
        if "revision" not in [
            r[1] for r in conn.execute("PRAGMA table_info(assessments)")
        ]:
            conn.execute("ALTER TABLE assessments ADD COLUMN revision TEXT")

        # Every existing ruleset becomes its own first, active revision.
        # Content is moved verbatim and never rewritten: the revision id
        # is the hash of the stored (already normalized) body.
        for rs in conn.execute(
            "SELECT name, body, created_at FROM rulesets"
        ).fetchall():
            rules = json.loads(rs["body"])
            rid = revision_id(rules)
            conn.execute(
                "INSERT OR IGNORE INTO ruleset_revisions"
                "(ruleset_name, revision, body, content_hash, base_revision, created_at)"
                " VALUES (?,?,?,?,NULL,?)",
                (rs["name"], rid, rs["body"], stable_hash(rules), rs["created_at"]),
            )
            conn.execute(
                "INSERT OR IGNORE INTO ruleset_active(ruleset_name, revision, activated_at)"
                " VALUES (?,?,?)",
                (rs["name"], rid, rs["created_at"]),
            )
            conn.execute(
                "INSERT OR IGNORE INTO ruleset_activations"
                "(ruleset_name, revision, previous_revision, activated_at)"
                " VALUES (?,?,NULL,?)",
                (rs["name"], rid, rs["created_at"]),
            )

        # Stamp assessments with the revision of the pinned snapshot when
        # the (ruleset, revision) pair exists.  Unverifiable rows stay NULL.
        for row in conn.execute(
            "SELECT id, ruleset, rules FROM assessments WHERE revision IS NULL AND rules IS NOT NULL"
        ).fetchall():
            if not row["ruleset"]:
                continue
            rules = json.loads(row["rules"])
            rid = revision_id(rules)
            hit = conn.execute(
                "SELECT 1 FROM ruleset_revisions WHERE ruleset_name = ? AND revision = ?",
                (row["ruleset"], rid),
            ).fetchone()
            if hit is not None:
                conn.execute(
                    "UPDATE assessments SET revision = ? WHERE id = ?", (rid, row["id"])
                )

    @staticmethod
    def _verified_legacy_rules(conn, row):
        """Return a legacy row's original rules only when they are provable.

        A snapshot is accepted when all of the following hold:

        1. the row named a stored ruleset (inline-rule requests were never
           persisted, so they can never be restored),
        2. a ruleset with that name still exists and its current, normalized
           content hashes to the assessment id stored on the row,
        3. re-running the assessment with those rules reproduces the stored
           ``summary_hash`` exactly.

        Anything else -- a deleted/overwritten ruleset, an inline-rule row,
        any hash mismatch -- yields None; callers must then refuse to mint a
        confirmation rather than inventing a replay.
        """
        name = row["ruleset"]
        if not name:
            return None
        rs = conn.execute("SELECT body FROM rulesets WHERE name = ?", (name,)).fetchone()
        if rs is None:
            return None
        rules = json.loads(rs["body"])
        request = json.loads(row["request"])
        if stable_hash({"request": request, "rules": rules})[:32] != row["id"]:
            return None
        stored = json.loads(row["result"])
        replayed = assess_normalized(request, rules)
        if replayed.get("summary_hash") != stored.get("summary_hash"):
            return None
        return rules

    def recover_legacy_rules(self, aid):
        """Best-effort recovery of a pre-upgrade assessment's rules.

        Re-runs verification on demand (the ruleset may have been restored
        after startup).  Backfills and returns the normalized snapshot on
        success, otherwise None -- never a guessed payload.
        """
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT id, ruleset, rules, request, result"
                    " FROM assessments WHERE id = ?",
                    (aid,),
                ).fetchone()
                if row is None:
                    return None
                if row["rules"] is not None:
                    return json.loads(row["rules"])
                rules = self._verified_legacy_rules(conn, row)
                if rules is not None:
                    rid = revision_id(rules)
                    hit = conn.execute(
                        "SELECT 1 FROM ruleset_revisions"
                        " WHERE ruleset_name = ? AND revision = ?",
                        (row["ruleset"], rid),
                    ).fetchone()
                    if hit is not None:
                        conn.execute(
                            "UPDATE assessments SET rules = ?, revision = ? WHERE id = ?",
                            (json.dumps(rules, sort_keys=True), rid, aid),
                        )
                    else:
                        conn.execute(
                            "UPDATE assessments SET rules = ? WHERE id = ?",
                            (json.dumps(rules, sort_keys=True), aid),
                        )
                    conn.commit()
                return rules
            finally:
                conn.close()

    # -- rulesets (legacy PUT/GET semantics, revision-backed) -------------

    def put_ruleset(self, name, rules):
        """Create or replace the *active* content (legacy PUT semantics).

        Backed by immutable revisions: new content appends a revision and
        atomically flips the active pointer; identical content is a no-op
        and destroys nothing.  Returns ``created`` (ruleset did not exist).
        """
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                active = self._active_row(conn, name)
                created = active is None
                body = json.dumps(rules, sort_keys=True)
                rid = revision_id(rules)
                if active is None:
                    # Bootstrap: legacy PUT publishes the first revision.  A
                    # never-activated draft of the same content may already
                    # own the revision row, so the insert is idempotent.
                    now = _now()
                    conn.execute(
                        "INSERT OR IGNORE INTO ruleset_revisions"
                        "(ruleset_name, revision, body, content_hash, base_revision, created_at)"
                        " VALUES (?,?,?,?,NULL,?)",
                        (name, rid, body, stable_hash(rules), now),
                    )
                    conn.execute(
                        "INSERT INTO ruleset_active(ruleset_name, revision, activated_at)"
                        " VALUES (?,?,?)",
                        (name, rid, now),
                    )
                    conn.execute(
                        "INSERT INTO ruleset_activations"
                        "(ruleset_name, revision, previous_revision, activated_at)"
                        " VALUES (?,?,NULL,?)",
                        (name, rid, now),
                    )
                elif active["revision"] != rid:
                    # Legacy PUT is unconditional: its base is the content
                    # currently active at the moment of the write.
                    self._append_and_activate(conn, name, active, rules, body, rid)
                conn.execute(
                    "INSERT OR REPLACE INTO rulesets(name, body, created_at) VALUES (?,?,?)",
                    (name, body, _now()),
                )
                conn.commit()
                return created
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def get_ruleset(self, name):
        """Active ruleset content (legacy GET mirror)."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT r.body AS body FROM ruleset_active a"
                " JOIN ruleset_revisions r"
                " ON r.ruleset_name = a.ruleset_name AND r.revision = a.revision"
                " WHERE a.ruleset_name = ?",
                (name,),
            ).fetchone()
            if row is None:
                row = conn.execute(
                    "SELECT body FROM rulesets WHERE name = ?", (name,)
                ).fetchone()
        finally:
            conn.close()
        return json.loads(row["body"]) if row else None

    def list_rulesets(self):
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT ruleset_name FROM ruleset_active ORDER BY ruleset_name"
            ).fetchall()
        finally:
            conn.close()
        return [r["ruleset_name"] for r in rows]

    # -- immutable revisions & atomic activation --------------------------

    @staticmethod
    def _active_row(conn, name):
        return conn.execute(
            "SELECT ruleset_name, revision, activated_at"
            " FROM ruleset_active WHERE ruleset_name = ?",
            (name,),
        ).fetchone()

    @staticmethod
    def _append_and_activate(conn, name, active, rules, body, rid, now=None):
        """Append (if new) a revision and switch the active pointer to it.

        Must run inside an already-open transaction; the active row read at
        transaction start is what gets recorded as ``previous_revision``.
        """
        if now is None:
            now = _now()
        conn.execute(
            "INSERT OR IGNORE INTO ruleset_revisions"
            "(ruleset_name, revision, body, content_hash, base_revision, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (name, rid, body, stable_hash(rules), active["revision"], now),
        )
        conn.execute(
            "UPDATE ruleset_active SET revision = ?, activated_at = ?"
            " WHERE ruleset_name = ? AND revision = ?",
            (rid, now, name, active["revision"]),
        )
        conn.execute(
            "INSERT INTO ruleset_activations"
            "(ruleset_name, revision, previous_revision, activated_at)"
            " VALUES (?,?,?,?)",
            (name, rid, active["revision"], now),
        )
        # Keep the legacy mirror in sync for old GET/PUT consumers.
        conn.execute(
            "INSERT OR REPLACE INTO rulesets(name, body, created_at) VALUES (?,?,?)",
            (name, body, now),
        )

    def create_ruleset_revision(self, name, rules, base_revision):
        """Create a content-deduplicated, immutable *draft* revision.

        A draft only appends to ``ruleset_revisions``: it never touches the
        active pointer, the activation audit or the legacy ``rulesets``
        mirror, so legacy GET and named accounting cannot observe it.  It
        becomes effective through a separate CAS activation.

        Optimistic concurrency: ``base_revision`` must equal the active
        revision as of the transaction start; for a not-yet-published
        ruleset it must be ``None`` (there is no active revision to base
        on).  An expired baseline raises ConflictError.
        """
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                active = self._active_row(conn, name)
                active_rev = active["revision"] if active is not None else None
                body = json.dumps(rules, sort_keys=True)
                rid = revision_id(rules)
                now = _now()
                if active_rev != base_revision:
                    raise ConflictError(
                        "stale base_revision for %r: active revision is %r"
                        % (name, active_rev),
                        {
                            "reason": "stale-base",
                            "base_revision": base_revision,
                            "active_revision": active_rev,
                        },
                    )
                existing = conn.execute(
                    "SELECT created_at, base_revision FROM ruleset_revisions"
                    " WHERE ruleset_name = ? AND revision = ?",
                    (name, rid),
                ).fetchone()
                if existing is None:
                    conn.execute(
                        "INSERT INTO ruleset_revisions"
                        "(ruleset_name, revision, body, content_hash, base_revision, created_at)"
                        " VALUES (?,?,?,?,?,?)",
                        (name, rid, body, stable_hash(rules), base_revision, now),
                    )
                    created, stored_base, stored_at = True, base_revision, now
                else:
                    # Identical normalized content collapses onto the same
                    # immutable revision row (a re-submitted draft included).
                    created = False
                    stored_base = existing["base_revision"]
                    stored_at = existing["created_at"]
                conn.commit()
                return {
                    "name": name,
                    "revision": rid,
                    "content_hash": stable_hash(rules),
                    "base_revision": stored_base,
                    "created_at": stored_at,
                    "active": rid == active_rev,
                    "created": created,
                }
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def activate_ruleset_revision(self, name, rid, expected_active):
        """Atomically switch (or set) the active revision via compare-and-set.

        The active revision is (re-)read inside one transaction and the
        switch, its audit row and the previous-revision provenance all
        commit together, along with the legacy ``rulesets`` mirror sync.
        A lost race (the active revision moved since the caller fetched
        it, or another transaction published the first revision while
        this one still expected ``None``) raises ConflictError and changes
        nothing.  Activating the revision already active is idempotent and
        adds no audit row.
        """
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                active = self._active_row(conn, name)
                current = active["revision"] if active is not None else None
                target = conn.execute(
                    "SELECT body FROM ruleset_revisions WHERE ruleset_name = ? AND revision = ?",
                    (name, rid),
                ).fetchone()
                if target is None:
                    # Distinguish a never-drafted name (404 ruleset) from a
                    # merely unknown revision of an existing ruleset.
                    any_rev = conn.execute(
                        "SELECT 1 FROM ruleset_revisions WHERE ruleset_name = ?",
                        (name,),
                    ).fetchone()
                    if any_rev is None:
                        raise NotFound("ruleset %r not found" % name)
                    raise NotFound(
                        "revision %r of ruleset %r not found" % (rid, name)
                    )
                if current != expected_active:
                    raise ConflictError(
                        "expected active revision %r but it is %r"
                        % (expected_active, current),
                        {
                            "reason": "active-moved",
                            "expected_active_revision": expected_active,
                            "active_revision": current,
                        },
                    )
                now = _now()
                if current is None:
                    # First publish: create the pointer and the first audit
                    # row, and mirror the content for legacy GET/PUT users.
                    conn.execute(
                        "INSERT INTO ruleset_active(ruleset_name, revision, activated_at)"
                        " VALUES (?,?,?)",
                        (name, rid, now),
                    )
                    conn.execute(
                        "INSERT INTO ruleset_activations"
                        "(ruleset_name, revision, previous_revision, activated_at)"
                        " VALUES (?,?,NULL,?)",
                        (name, rid, now),
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO rulesets(name, body, created_at)"
                        " VALUES (?,?,?)",
                        (name, target["body"], now),
                    )
                    conn.commit()
                    return {
                        "name": name,
                        "revision": rid,
                        "active_revision": rid,
                        "activated": True,
                        "previous_revision": None,
                        "activated_at": now,
                    }
                if current == rid:
                    result = {
                        "name": name,
                        "revision": rid,
                        "active_revision": rid,
                        "activated": False,
                        "previous_revision": rid,
                        "activated_at": active["activated_at"],
                    }
                    conn.commit()
                    return result
                conn.execute(
                    "UPDATE ruleset_active SET revision = ?, activated_at = ?"
                    " WHERE ruleset_name = ? AND revision = ?",
                    (rid, now, name, current),
                )
                conn.execute(
                    "INSERT INTO ruleset_activations"
                    "(ruleset_name, revision, previous_revision, activated_at)"
                    " VALUES (?,?,?,?)",
                    (name, rid, current, now),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO rulesets(name, body, created_at)"
                    " VALUES (?,?,?)",
                    (name, target["body"], now),
                )
                conn.commit()
                return {
                    "name": name,
                    "revision": rid,
                    "active_revision": rid,
                    "activated": True,
                    "previous_revision": current,
                    "activated_at": now,
                }
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def list_ruleset_revisions(self, name):
        """Return the immutable revision history plus activation records.

        A ruleset that has drafts but no published revision is still
        returned (``active_revision`` is ``None`` and the activation
        chain is empty); only a name with no revisions at all yields
        ``None``.
        """
        conn = self._connect()
        try:
            active = self._active_row(conn, name)
            active_rev = active["revision"] if active is not None else None
            revs = conn.execute(
                "SELECT revision, body, content_hash, base_revision, created_at"
                " FROM ruleset_revisions WHERE ruleset_name = ?"
                " ORDER BY seq",
                (name,),
            ).fetchall()
            if not revs:
                return None
            activations = conn.execute(
                "SELECT revision, previous_revision, activated_at"
                " FROM ruleset_activations WHERE ruleset_name = ?"
                " ORDER BY seq",
                (name,),
            ).fetchall()
        finally:
            conn.close()
        return {
            "name": name,
            "active_revision": active_rev,
            "revisions": [
                {
                    "revision": r["revision"],
                    "content_hash": r["content_hash"],
                    "base_revision": r["base_revision"],
                    "active": r["revision"] == active_rev,
                    "created_at": r["created_at"],
                    "ruleset": json.loads(r["body"]),
                }
                for r in revs
            ],
            "activations": [
                {
                    "revision": a["revision"],
                    "previous_revision": a["previous_revision"],
                    "activated_at": a["activated_at"],
                }
                for a in activations
            ],
        }

    # -- assessments ------------------------------------------------------

    def put_assessment(self, aid, ruleset, rules, request, result, revision=None):
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO assessments"
                    "(id, ruleset, rules, revision, request, result, status,"
                    " summary_hash, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        aid,
                        ruleset,
                        json.dumps(rules, sort_keys=True),
                        revision,
                        json.dumps(request, sort_keys=True),
                        json.dumps(result, sort_keys=True),
                        result.get("status", ""),
                        result.get("summary_hash", ""),
                        _now(),
                    ),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def create_named_assessment(self, ruleset_name, norm):
        """Evaluate against the active revision inside one transaction.

        The active revision, its ruleset body and the audit-relevant
        revision id are read at transaction start, the assessment is
        evaluated against that snapshot and the row (carrying the exact
        revision id it ran with) is written in the same transaction, so
        the recorded ``revision`` is provably the one that was active when
        the accounting began -- an activation racing with the assessment
        cannot make the row claim a different revision.
        """
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                active = conn.execute(
                    "SELECT revision FROM ruleset_active WHERE ruleset_name = ?",
                    (ruleset_name,),
                ).fetchone()
                if active is None:
                    raise NotFound("ruleset %r not found" % ruleset_name)
                rev_row = conn.execute(
                    "SELECT body FROM ruleset_revisions"
                    " WHERE ruleset_name = ? AND revision = ?",
                    (ruleset_name, active["revision"]),
                ).fetchone()
                rules = json.loads(rev_row["body"])
                aid = stable_hash({"request": norm, "rules": rules})[:32]
                row = conn.execute(
                    "SELECT result FROM assessments WHERE id = ?", (aid,)
                ).fetchone()
                if row is not None:
                    conn.commit()
                    return {
                        "id": aid,
                        "created": False,
                        "result": json.loads(row["result"]),
                        "revision": active["revision"],
                    }
                result = assess_normalized(norm, rules)
                conn.execute(
                    "INSERT INTO assessments"
                    "(id, ruleset, rules, revision, request, result, status,"
                    " summary_hash, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        aid,
                        ruleset_name,
                        json.dumps(rules, sort_keys=True),
                        active["revision"],
                        json.dumps(norm, sort_keys=True),
                        json.dumps(result, sort_keys=True),
                        result.get("status", ""),
                        result.get("summary_hash", ""),
                        _now(),
                    ),
                )
                conn.commit()
                return {
                    "id": aid,
                    "created": True,
                    "result": result,
                    "revision": active["revision"],
                }
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def get_assessment(self, aid):
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM assessments WHERE id = ?", (aid,)).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return {
            "id": row["id"],
            "ruleset": row["ruleset"],
            "rules": json.loads(row["rules"]) if row["rules"] is not None else None,
            "revision": row["revision"],
            "request": json.loads(row["request"]),
            "result": json.loads(row["result"]),
            "created_at": row["created_at"],
        }

    def review_impact_inputs(self, name, target_revision, base_revision, ids):
        """Validate and snapshot one pre-publication impact-review batch.

        Everything stateful is confirmed inside a *single read
        transaction* (a deferred WAL transaction gives one consistent
        snapshot for every SELECT):

        1. the target revision exists as immutable content of *name*;
        2. the active revision, re-read inside the transaction, still
           equals ``base_revision`` (the revision the caller read when
           it created the draft);
        3. every requested assessment exists, belongs to *name* and
           recorded that active revision.

        Any failure raises ConflictError -- the whole review is refused,
        no partial report is possible.  On success it returns
        ``(target_rules, rows)`` where each row carries the assessment's
        pinned normalized request and persisted result; the caller closes
        the transaction (by returning) and only then runs the (CPU-bound)
        re-evaluation.
        """
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            active = self._active_row(conn, name)
            active_rev = active["revision"] if active is not None else None

            target = conn.execute(
                "SELECT body FROM ruleset_revisions"
                " WHERE ruleset_name = ? AND revision = ?",
                (name, target_revision),
            ).fetchone()
            if target is None:
                any_rev = conn.execute(
                    "SELECT 1 FROM ruleset_revisions WHERE ruleset_name = ?",
                    (name,),
                ).fetchone()
                if any_rev is None:
                    raise ConflictError(
                        "ruleset %r has no revisions; target revision %r cannot be "
                        "reviewed" % (name, target_revision),
                        {"reason": "ruleset-not-found", "ruleset": name},
                    )
                raise ConflictError(
                    "target revision %r of ruleset %r not found"
                    % (target_revision, name),
                    {
                        "reason": "unknown-target-revision",
                        "target_revision": target_revision,
                        "active_revision": active_rev,
                    },
                )

            if active_rev != base_revision:
                raise ConflictError(
                    "stale base revision for impact review of %r: active revision "
                    "is %r, review was prepared against %r"
                    % (name, active_rev, base_revision),
                    {
                        "reason": "active-moved",
                        "base_revision": base_revision,
                        "active_revision": active_rev,
                    },
                )

            placeholders = ",".join("?" for _ in ids)
            found = {
                r["id"]: r
                for r in conn.execute(
                    "SELECT id, ruleset, revision, request, result"
                    " FROM assessments WHERE id IN (%s)" % placeholders,
                    list(ids),
                ).fetchall()
            }
            missing = sorted(aid for aid in ids if aid not in found)
            if missing:
                raise ConflictError(
                    "impact review references unknown assessment ids: %s"
                    % ", ".join(missing),
                    {"reason": "assessment-not-found", "assessment_ids": missing},
                )

            foreign = sorted(
                aid for aid in ids if found[aid]["ruleset"] != name
            )
            if foreign:
                raise ConflictError(
                    "impact review accepts only assessments of ruleset %r; these "
                    "ids belong to another ruleset (or inline rules): %s"
                    % (name, ", ".join(foreign)),
                    {
                        "reason": "assessment-ruleset-mismatch",
                        "ruleset": name,
                        "assessment_ids": foreign,
                    },
                )

            mismatched = sorted(
                aid
                for aid in ids
                if found[aid]["revision"] != active_rev
            )
            if mismatched:
                raise ConflictError(
                    "impact review accepts only assessments recorded against the "
                    "active revision %r of ruleset %r; these ids record another "
                    "revision: %s"
                    % (active_rev, name, ", ".join(mismatched)),
                    {
                        "reason": "assessment-revision-mismatch",
                        "active_revision": active_rev,
                        "assessment_ids": mismatched,
                    },
                )

            target_rules = json.loads(target["body"])
            rows = [
                {
                    "id": found[aid]["id"],
                    "request": json.loads(found[aid]["request"]),
                    "result": json.loads(found[aid]["result"]),
                }
                for aid in ids
            ]
            return target_rules, rows
        finally:
            conn.rollback()
            conn.close()

    # -- confirmations ----------------------------------------------------

    def put_confirmation(self, aid, body):
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO confirmations(assessment_id, body, created_at)"
                    " VALUES (?,?,?)",
                    (aid, json.dumps(body, sort_keys=True), _now()),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def get_confirmation(self, aid):
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT body FROM confirmations WHERE assessment_id = ?", (aid,)
            ).fetchone()
        finally:
            conn.close()
        return json.loads(row["body"]) if row else None

    # -- release batches ---------------------------------------------------

    @staticmethod
    def _snapshot_rules_within(conn, row):
        """Prove and return an assessment's rules snapshot inside a tx.

        Rows that already carry a pinned ``rules`` snapshot are accepted
        as-is by the caller; this covers rows without one (pre-upgrade
        databases): the original rules are returned only when they can be
        *proven* -- either the named revision the assessment recorded still
        exists as immutable content, or the legacy ruleset mirror content
        hashes to the assessment id and a replay reproduces the stored
        ``summary_hash``.  Anything else yields ``None``; the caller must
        refuse the batch rather than registering against guessed rules.
        """
        request = json.loads(row["request"])
        stored = json.loads(row["result"])
        name = row["ruleset"]
        if name and row["revision"]:
            rev_row = conn.execute(
                "SELECT body FROM ruleset_revisions"
                " WHERE ruleset_name = ? AND revision = ?",
                (name, row["revision"]),
            ).fetchone()
            if rev_row is not None:
                rules = json.loads(rev_row["body"])
                if stable_hash({"request": request, "rules": rules})[:32] == row["id"]:
                    replayed = assess_normalized(request, rules)
                    if replayed.get("summary_hash") == stored.get("summary_hash"):
                        return rules
        rules = Store._verified_legacy_rules(conn, row)
        return rules

    def register_release_batch(self, batch_id, entries):
        """Register an immutable release batch in a single transaction.

        Everything stateful happens under one ``BEGIN IMMEDIATE``
        transaction:

        * every named assessment must exist and be ``status == "ok"``;
        * the exact immutable rule revision each assessment recorded is
          read and locked inside the transaction (a named assessment whose
          snapshot cannot be tied back to an existing immutable revision is
          refused with ``rules-snapshot-unverifiable``; inline-rule
          assessments carry their own pinned snapshot);
        * each entry's declared artifacts/versions/dependencies are
          verified against the assessment's pinned request, and the batch
          is merged (shared components deduplicated; inconsistent name /
          version / adopted license across assessments refuses the whole
          batch with 422).

        Only after every precondition holds are the manifest and the full
        per-assessment rule snapshots written -- all in that same
        transaction, so a failure writes no partial records.  Reusing a
        ``batch_id`` with equal rebuilt content is idempotent (the frozen
        manifest is returned); a reused id with different content fails
        with 409 and never overwrites the registered snapshot.
        """
        from .batches import build_batch_manifest, verify_entry

        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                existing = conn.execute(
                    "SELECT manifest, batch_hash, created_at"
                    " FROM release_batches WHERE batch_id = ?",
                    (batch_id,),
                ).fetchone()

                ids = [e["assessment_id"] for e in entries]
                placeholders = ",".join("?" for _ in ids)
                found = {
                    r["id"]: r
                    for r in conn.execute(
                        "SELECT id, ruleset, rules, revision, request, result"
                        " FROM assessments WHERE id IN (%s)" % placeholders,
                        ids,
                    ).fetchall()
                }
                missing = sorted(aid for aid in ids if aid not in found)
                if missing:
                    raise ConflictError(
                        "release batch references unknown assessment ids: %s"
                        % ", ".join(missing),
                        {"reason": "assessment-not-found", "assessment_ids": missing},
                    )

                items = []
                for entry in sorted(entries, key=lambda e: e["assessment_id"]):
                    aid = entry["assessment_id"]
                    row = found[aid]
                    request = json.loads(row["request"])
                    result = json.loads(row["result"])
                    status = result.get("status")
                    if status != "ok":
                        raise ConflictError(
                            "assessment %r has status %r; only 'ok' assessments"
                            " can be registered in a release batch"
                            % (aid, status),
                            {
                                "reason": "assessment-not-ok",
                                "assessment_id": aid,
                                "status": status,
                            },
                        )

                    rules = (
                        json.loads(row["rules"]) if row["rules"] is not None else None
                    )
                    ruleset = row["ruleset"]
                    revision = row["revision"]
                    if rules is None:
                        rules = self._snapshot_rules_within(conn, row)
                        if rules is None:
                            raise ConflictError(
                                "assessment %r: the rule revision it was recorded"
                                " against can no longer be verified" % aid,
                                {
                                    "reason": "rules-snapshot-unverifiable",
                                    "assessment_id": aid,
                                },
                            )
                    if ruleset:
                        # Read and lock the immutable rule revision inside
                        # this transaction; the pinned snapshot must be
                        # byte-identical (same content hash) to that
                        # revision's frozen body.
                        rev = revision or revision_id(rules)
                        rev_row = conn.execute(
                            "SELECT body FROM ruleset_revisions"
                            " WHERE ruleset_name = ? AND revision = ?",
                            (ruleset, rev),
                        ).fetchone()
                        if rev_row is None or stable_hash(
                            json.loads(rev_row["body"])
                        ) != stable_hash(rules):
                            raise ConflictError(
                                "assessment %r: ruleset %r revision %r cannot be"
                                " verified against an immutable revision"
                                % (aid, ruleset, rev),
                                {
                                    "reason": "rules-snapshot-unverifiable",
                                    "assessment_id": aid,
                                    "ruleset": ruleset,
                                    "revision": rev,
                                },
                            )
                        revision = rev

                    verify_entry(entry, request, result)
                    items.append(
                        {
                            "entry": entry,
                            "request": request,
                            "result": result,
                            "ruleset": ruleset,
                            "revision": revision,
                            "rules": rules,
                        }
                    )

                manifest = build_batch_manifest(batch_id, items)

                if existing is not None:
                    if existing["batch_hash"] != manifest["batch_hash"]:
                        raise ConflictError(
                            "batch_id %r is already registered with different"
                            " content" % batch_id,
                            {
                                "reason": "batch-id-conflict",
                                "batch_id": batch_id,
                                "existing_batch_hash": existing["batch_hash"],
                                "requested_batch_hash": manifest["batch_hash"],
                            },
                        )
                    conn.commit()
                    return {
                        "manifest": json.loads(existing["manifest"]),
                        "created": False,
                        "created_at": existing["created_at"],
                    }

                now = _now()
                conn.execute(
                    "INSERT INTO release_batches"
                    "(batch_id, manifest, batch_hash, created, created_at)"
                    " VALUES (?,?,?,?,?)",
                    (
                        batch_id,
                        json.dumps(manifest, sort_keys=True),
                        manifest["batch_hash"],
                        1,
                        now,
                    ),
                )
                for seq, item in enumerate(
                    sorted(items, key=lambda it: it["entry"]["assessment_id"])
                ):
                    conn.execute(
                        "INSERT INTO release_batch_assessments"
                        "(batch_id, assessment_id, ruleset, revision, rules, seq)"
                        " VALUES (?,?,?,?,?,?)",
                        (
                            batch_id,
                            item["entry"]["assessment_id"],
                            item["ruleset"],
                            item["revision"],
                            json.dumps(item["rules"], sort_keys=True),
                            seq,
                        ),
                    )
                conn.commit()
                return {"manifest": manifest, "created": True, "created_at": now}
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def get_release_batch(self, batch_id):
        """Return the frozen manifest of a registered batch, else None.

        The manifest is read verbatim from the registration row -- later
        ruleset publications or assessment changes cannot alter it.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT manifest, created_at FROM release_batches WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return {"manifest": json.loads(row["manifest"]), "created_at": row["created_at"]}
