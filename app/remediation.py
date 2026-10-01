"""Component license remediation planning for ``status == "conflict"``
assessments.

For every adjustable component the caller submits candidate license
expressions, each with a stable id and a non-negative cost; a component
may always keep its original expression at zero cost (the implicit
``keep`` candidate).  The service then solves *globally* -- never
component-by-component greedy, never a fixed cut-off -- over every joint
candidate combination:

* a combination is releasable only when re-running the assessment with
  the substituted expressions yields ``status == "ok"`` against the
  exact normalized request and rules snapshot persisted with the
  assessment -- never the currently active ruleset;
* releasable combinations are ordered by
  ``(total cost, changed-component count, component-id/candidate-id
  sequence)`` -- the sequence lists the *changed* picks as
  ``(component id, candidate id)`` pairs in component-id order (kept
  components, like skipped items in the other planners, are absent) --
  so the optimum is unique;
* when nothing is releasable, the same lexicographic spirit picks a
  unique best failing combination --
  ``(conflicting-component count, total cost, changed count, sequence)``
  -- returned with HTTP 409 and the recomputed conflict explanation.

Candidate legality (unparseable expression, unknown license/exception in
the pinned snapshot, bad/negative cost, duplicate or unknown component
reference) fails the whole request with HTTP 422 -- no partial plan is
ever produced.

Costs are exact rational numbers internally (mirroring the
responsibility planner), so floating-point rounding can never disturb
the optimization or a tie-break; responses render costs as JSON numbers
(integers when integral).
"""

import math
from fractions import Fraction

from .common import ClientError, ConflictError, stable_hash
from .expression import ExpressionError, exception_ids, license_ids, parse
from .fulfillment import UnprocessableError
from .solver import assess_normalized

# Combination-space cap for exhaustive enumeration: the product over
# components of (submitted candidates + 1, the implicit keep).  Larger
# spaces use the complete branch-and-bound path.
MAX_REMEDIATION_COMBINATIONS = 4096

# Stable id of the implicit zero-cost "keep the original expression"
# candidate.  It is an actual option of every component but never enters
# the changed-component sequence (it is precisely the no-op choice).
KEEP_ID = "keep"


class RemediationConflict(ConflictError):
    """HTTP 409 carrying the structured best-failure-plan payload."""

    def __init__(self, payload):
        super().__init__(payload["error"])
        self.payload = payload


# ---------------------------------------------------------------------------
# validation / normalization
# ---------------------------------------------------------------------------

def _cost(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnprocessableError("%s.cost must be a non-negative number" % where)
    if not math.isfinite(value) or value < 0:
        raise UnprocessableError("%s.cost must be a non-negative number" % where)
    return Fraction(value)


def normalize_candidates(body, norm, rules):
    """Validate the submitted component -> candidates mapping.

    Returns ``{component_id: [candidate, ...]}`` with every component of
    the assessment present (submitted candidates first, sorted by id,
    then the implicit zero-cost keep candidate).  Candidates are sorted
    and costs are exact rationals, so neither the request's component
    order nor its candidate order can influence the result.

    Any illegal candidate or reference to an unknown component raises
    :class:`UnprocessableError` (HTTP 422) -- the request fails as a
    whole, before any conclusion is computed.
    """
    if not isinstance(body, dict):
        raise TypeError("request body must be a JSON object")
    raw_map = body.get("components")
    if not isinstance(raw_map, dict):
        raise TypeError("remediation request requires a 'components' object")

    license_rules = rules["licenses"]
    exception_rules = rules.get("exceptions", {})
    originals = {c["id"]: c for c in norm["components"]}

    submissions = {}
    for cid, raw_cands in raw_map.items():
        if not isinstance(cid, str) or cid not in originals:
            raise UnprocessableError(
                "candidate submission references unknown component %r" % cid
            )
        if not isinstance(raw_cands, list) or not raw_cands:
            raise UnprocessableError(
                "components[%r] must be a non-empty array of candidates" % cid
            )
        where = "components[%r]" % cid
        seen = set()
        cands = []
        for i, raw in enumerate(raw_cands):
            cwhere = "%s[%d]" % (where, i)
            if not isinstance(raw, dict):
                raise UnprocessableError("%s must be an object" % cwhere)
            cand_id = raw.get("id")
            if not isinstance(cand_id, str) or not cand_id.strip():
                raise UnprocessableError("%s.id must be a non-empty string" % cwhere)
            cand_id = cand_id.strip()
            if cand_id in seen or cand_id == KEEP_ID:
                raise UnprocessableError(
                    "%s: duplicate or reserved candidate id %r" % (cwhere, cand_id)
                )
            seen.add(cand_id)
            expression = raw.get("expression")
            if not isinstance(expression, str) or not expression.strip():
                raise UnprocessableError(
                    "%s.expression must be a non-empty license expression string"
                    % cwhere
                )
            expression = expression.strip()
            try:
                ast = parse(expression)
            except ExpressionError as exc:
                raise UnprocessableError(
                    "%s.expression %r is not a valid license expression: %s"
                    % (cwhere, expression, exc.message)
                )
            unknown_licenses = [
                lid for lid in license_ids(ast) if lid not in license_rules
            ]
            unknown_exceptions = [
                eid for eid in exception_ids(ast) if eid not in exception_rules
            ]
            if unknown_licenses or unknown_exceptions:
                raise UnprocessableError(
                    "%s.expression %r references ids absent from the assessment's "
                    "pinned rules snapshot: %s"
                    % (
                        cwhere,
                        expression,
                        sorted(set(unknown_licenses) | set(unknown_exceptions)),
                    )
                )
            cost = _cost(raw.get("cost", 0), cwhere)
            cands.append(
                {
                    "id": cand_id,
                    "expression": expression,
                    "cost": cost,
                    "changed": expression != originals[cid]["license"],
                }
            )
        cands.sort(key=lambda c: c["id"])
        submissions[cid] = cands

    # Every component gets the implicit zero-cost keep option, submitted
    # candidates first in id order.
    choices = {}
    for cid in sorted(originals):
        options = list(submissions.get(cid, ()))
        options.append(
            {
                "id": KEEP_ID,
                "expression": originals[cid]["license"],
                "cost": Fraction(0),
                "changed": False,
            }
        )
        choices[cid] = options
    return choices


# ---------------------------------------------------------------------------
# global optimization
# ---------------------------------------------------------------------------

def _num(frac):
    """Render an exact rational cost as a JSON number (int when integral)."""
    if frac.denominator == 1:
        return frac.numerator
    return float(frac)


def _solve(norm, rules, choices, scale):
    """Complete search over every joint candidate combination.

    Returns ``(mode, best_release, best_failure)`` where each best is
    ``(key, picked, result)``:

    * release key: ``(scaled cost, changed count, sequence)``
    * failure key: ``(conflicting-component count, scaled cost, changed
      count, sequence)``
    * ``sequence`` is the tuple of ``(component id, candidate id)`` pairs
      of the *changed* picks, in component-id order (unchanged keeps
      contribute nothing -- the same convention the responsibility and
      fulfillment planners use for skipped/unassigned items);
    * ``picked`` is the chosen option index per component (component-id
      order), ``result`` the re-assessment of the substituted request.

    The traversal is a complete iterative DFS in both modes; the
    branch-and-bound path only prunes subtrees whose lexicographic lower
    bound cannot improve the incumbent (remaining components add zero
    cost / zero change / no sequence entry at best when kept), so both
    paths return the identical optimum.
    """
    comp_ids = sorted(choices)
    options = [choices[cid] for cid in comp_ids]
    total = 1
    for opts in options:
        total *= len(opts)
        if total > MAX_REMEDIATION_COMBINATIONS:
            break
    mode = "exhaustive" if total <= MAX_REMEDIATION_COMBINATIONS else "branch-and-bound"

    def scaled(cost):
        return cost.numerator * (scale // cost.denominator)

    # Per-option key parts: (candidate id, scaled cost, changed).
    # Submitted options stay in id order; the implicit keep is only ever
    # a zero-cost no-op (changed == 0) and never enters the sequence.
    opt_parts = []
    for cid, opts in zip(comp_ids, options):
        parts = []
        for opt in opts:
            parts.append(
                (opt["id"], scaled(opt["cost"]), int(opt["changed"]))
            )
        opt_parts.append(parts)

    n = len(comp_ids)
    base_components = norm["components"]
    index_of = {cid: k for k, cid in enumerate(comp_ids)}
    expressions_by_comp = [
        [opt["expression"] for opt in opts] for opts in options
    ]

    def reassess(picked):
        # Rebuild a shallow normalized request with the substituted
        # component licenses; every leaf cost is just the re-assessment.
        components = []
        for comp in base_components:
            k = index_of[comp["id"]]
            components.append(
                {
                    "id": comp["id"],
                    "name": comp["name"],
                    "version": comp["version"],
                    "license": expressions_by_comp[k][picked[k]],
                    "linkage": comp["linkage"],
                }
            )
        modified = dict(norm)
        modified["components"] = components
        return assess_normalized(modified, rules)

    # Iterative DFS.  Invariant: while a frame [depth, pointer] is on the
    # stack, cost_now/changed_now/seq reflect the chosen option of every
    # component *above* depth; the option chosen for depth is applied
    # when descending and undone when the child frame returns.
    best_release = None   # (key, picked, result)
    best_failure = None
    picked = [-1] * n
    cost_now = 0
    changed_now = 0
    seq = []

    def undo_level(d):
        nonlocal cost_now, changed_now
        _cid, scost, changed = opt_parts[d][picked[d]]
        cost_now -= scost
        changed_now -= changed
        if changed:
            seq.pop()
        picked[d] = -1

    stack = [[0, 0]]
    while stack:
        depth, _ = stack[-1]
        if depth == n:
            result = reassess(picked)
            sequence = tuple(seq)
            if result.get("status") == "ok":
                key = (cost_now, changed_now, sequence)
                if best_release is None or key < best_release[0]:
                    best_release = (key, list(picked), result)
            else:
                n_conflicts = len(result.get("conflicting_components", []))
                key = (n_conflicts, cost_now, changed_now, sequence)
                if best_failure is None or key < best_failure[0]:
                    best_failure = (key, list(picked), result)
            stack.pop()
            if depth > 0:
                undo_level(depth - 1)
            continue

        j = stack[-1][1]
        if j >= len(opt_parts[depth]):
            stack.pop()
            if depth > 0:
                undo_level(depth - 1)
            continue

        stack[-1][1] += 1
        cand_id, scost, changed = opt_parts[depth][j]
        prefix = tuple(seq)
        if changed:
            prefix = prefix + ((comp_ids[depth], cand_id),)

        if mode == "branch-and-bound":
            # Admissible lexicographic lower bounds for every completion
            # of this prefix: remaining components add zero cost / zero
            # change / no sequence entry at best (all kept).  The
            # exhaustive path is a pruning-free reference oracle.
            if best_release is not None and (
                cost_now + scost, changed_now + changed, prefix
            ) >= best_release[0]:
                continue
            if best_failure is not None and (
                # Zero remaining conflicts is always the conflict bound.
                0, cost_now + scost, changed_now + changed, prefix
            ) >= best_failure[0]:
                continue

        picked[depth] = j
        cost_now += scost
        changed_now += changed
        if changed:
            seq.append((comp_ids[depth], cand_id))
        stack.append([depth + 1, 0])

    return mode, best_release, best_failure


# ---------------------------------------------------------------------------
# response assembly
# ---------------------------------------------------------------------------

def _build_payload(assessment_id, norm, choices, picked, mode, result, release):
    comp_ids = sorted(choices)
    options = [choices[cid] for cid in comp_ids]
    originals = {c["id"]: c["license"] for c in norm["components"]}

    selection = []
    replacements = []
    total = Fraction(0)
    for k, cid in enumerate(comp_ids):
        opt = options[k][picked[k]]
        total += opt["cost"]
        entry = {
            "component": cid,
            "candidate_id": opt["id"],
            "original_expression": originals[cid],
            "expression": opt["expression"],
            "cost": _num(opt["cost"]),
            "changed": bool(opt["changed"]),
        }
        selection.append(entry)
        if opt["changed"]:
            replacements.append(
                {
                    "component": cid,
                    "candidate_id": opt["id"],
                    "from_expression": originals[cid],
                    "expression": opt["expression"],
                    "cost": _num(opt["cost"]),
                }
            )

    payload = {
        "assessment_id": assessment_id,
        "release": release,
        "selection_mode": mode,
        "total_cost": _num(total),
        "changed_components": len(replacements),
        "replacements": replacements,
        "selection": selection,
        "distribution": result.get("distribution", norm.get("distribution")),
    }
    if release:
        payload["artifact_obligations"] = result.get("artifact_obligations", [])
        payload["adopted"] = result.get("adopted", {})
        payload["components"] = result.get("components", [])
        payload["reassessed"] = {
            "status": result.get("status"),
            "release": result.get("release"),
            "summary_hash": result.get("summary_hash"),
        }
    else:
        payload["ready"] = False
        payload["reassessed"] = {
            "status": result.get("status"),
            "release": result.get("release"),
            "summary_hash": result.get("summary_hash"),
        }
        payload["conflicts"] = result.get("conflicts", [])
        payload["conflicting_components"] = result.get("conflicting_components", [])
        payload["confirmed_subgraph"] = result.get("confirmed_subgraph", {"components": []})
    return payload


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def plan_remediation(row, body, assessment_id):
    """Plan component license remediation for one persisted conflict.

    The caller guarantees the assessment exists; the pinned normalized
    request and rules snapshot are both verified here -- re-running the
    original assessment with its own snapshot must reproduce the stored
    ``summary_hash``.  An unverifiable snapshot, or a non-``conflict``
    assessment, raises :class:`ConflictError` (HTTP 409); current active
    rules are never substituted.
    """
    rules = row["rules"]
    norm = row["request"]
    stored = row["result"]
    if rules is None:
        raise ConflictError(
            "cannot plan remediation for assessment %r: its rules snapshot "
            "can no longer be verified" % assessment_id,
            {"reason": "rules-snapshot-unverifiable"},
        )
    replayed = assess_normalized(norm, rules)
    if replayed.get("summary_hash") != stored.get("summary_hash"):
        raise ConflictError(
            "cannot plan remediation for assessment %r: the persisted request "
            "and rules snapshot do not reproduce the stored result" % assessment_id,
            {"reason": "rules-snapshot-unverifiable"},
        )
    if stored.get("status") != "conflict":
        raise ConflictError(
            "assessment status is %r; remediation planning requires status "
            "'conflict'" % stored.get("status"),
            {"reason": "not-conflict", "status": stored.get("status")},
        )

    try:
        choices = normalize_candidates(body, norm, rules)
    except TypeError as exc:
        # Envelope-level malformed payload -> 400 in the API layer.
        raise ClientError(str(exc))

    # Exact integer cost scale shared by every comparison (lcm of
    # denominators, mirroring the responsibility planner).
    scale = 1
    for opts in choices.values():
        for opt in opts:
            d = opt["cost"].denominator
            scale = scale * d // math.gcd(scale, d)

    mode, best_release, best_failure = _solve(norm, rules, choices, scale)

    if best_release is not None:
        _key, picked, result = best_release
        payload = _build_payload(assessment_id, norm, choices, picked, mode, result, True)
        payload["remediation_hash"] = stable_hash(payload)
        return payload

    # No candidate combination makes the assessment releasable: return
    # the unique best failing plan with its recomputed conflict
    # explanation as a structured 409.
    _key, picked, result = best_failure
    payload = _build_payload(assessment_id, norm, choices, picked, mode, result, False)
    payload["remediation_hash"] = stable_hash(payload)
    payload["error"] = (
        "no candidate combination makes assessment %r releasable; best failing "
        "plan leaves %d conflicting component(s): %s"
        % (
            assessment_id,
            len(payload["conflicting_components"]),
            ", ".join(payload["conflicting_components"]) or "(none)",
        )
    )
    raise RemediationConflict(payload)
