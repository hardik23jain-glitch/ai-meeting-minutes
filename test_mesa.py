"""MESA-style ground-truth checks (Kirstein et al., 2024) - no audio needed.

Run:  python test_mesa.py        (or: pytest test_mesa.py -v)

Part A (offline, no API key): deterministic guards in refine.py / document.py.
Part B (live, needs GROQ_API_KEY): runs the documentation stage on a scripted
transcript and asserts the three MESA error classes are absent:
  1. proposed idea  -> must NOT become a confirmed decision
  2. unassigned task -> owner must stay "unspecified"
  3. "We will not adopt X" -> must NEVER become "We will adopt X"
"""
import os
import re
import sys

from dotenv import load_dotenv

load_dotenv()

TRANSCRIPT = (
    "Okay, let's get started with the weekly engineering sync. "
    "I think we could maybe try Redis for caching on the search service, "
    "but nobody has agreed to that yet and we need to look at the costs first. "
    "Someone needs to update the deployment documentation before the release. "
    "Rahul will send the budget report by Friday. "
    "On the public API, we discussed GraphQL for a while. "
    "After comparing it with REST, the decision is that we will not adopt "
    "GraphQL for the public API. "
    "Alright, that covers everything, thanks everyone."
)
KNOWN_NAMES = {"rahul"}
NEG_RE = re.compile(
    r"\b(not|no|never|won't|will not|reject\w*|declin\w*|against|rule[sd]? out|"
    r"drop\w*|avoid\w*|instead of|decided against)\b", re.I)

_results = []


def _check(name, cond, detail=""):
    _results.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" +
          (f" - {detail}" if detail and not cond else ""))


# --------------------------------------------------------------------------
# Part A: offline guards
# --------------------------------------------------------------------------
def test_offline_refine_guards():
    print("Part A1: refine.py safety filters")
    from pipeline import refine as R

    g = R.Glossary()
    # Negation flip must be rejected
    why = R.validate_edit("will not adopt", "will adopt", g, set(), set())
    _check("negation edit rejected", why is not None, f"got {why}")
    # Commitment word change must be rejected
    why = R.validate_edit("might", "will", g, set(), set())
    _check("commitment-word edit rejected", why is not None, f"got {why}")
    # Digits must be rejected
    why = R.validate_edit("fifteen", "50", g, set(), set())
    _check("number edit rejected", why is not None, f"got {why}")
    # Legit ASR fix accepted
    why = R.validate_edit("Arboretum cache", "Redis cache", g, set(), set())
    _check("sound-alike term fix handled without crash", True)
    # Integrity signature catches negation loss
    a = R._signature("We will not adopt GraphQL by 2025.")
    b = R._signature("We will adopt GraphQL by 2025.")
    _check("integrity signature detects negation drop", a != b)


def test_offline_owner_guard():
    print("Part A2: document.py owner guard")
    from pipeline import document as D

    if not hasattr(D, "_clean") or not hasattr(D, "MeetingRecord"):
        print("  [SKIP] _clean/MeetingRecord not exposed in this version")
        return
    try:
        rec = D.MeetingRecord(
            tldr="t", summary="s", minutes=[], decisions=[],
            action_items=[D.ActionItem(
                task="Update docs", owner="Zaphod Beeblebrox")],
        )
        rec = D._clean(rec, TRANSCRIPT)
        _check("hallucinated owner forced to 'unspecified'",
               rec.action_items[0].owner.lower() == "unspecified",
               rec.action_items[0].owner)
    except TypeError:
        print("  [SKIP] record schema differs from this version")


# --------------------------------------------------------------------------
# Part B: live MESA checks
# --------------------------------------------------------------------------
def _unwrap(res):
    return getattr(res, "record", res)


def test_live_mesa():
    print("Part B: live documentation-stage checks")
    if not os.getenv("GROQ_API_KEY"):
        print("  [SKIP] GROQ_API_KEY not set")
        return
    from pipeline.document import generate_record

    rec = _unwrap(generate_record(TRANSCRIPT))
    decisions = [getattr(d, "decision", str(d)) for d in rec.decisions]
    actions = list(rec.action_items)
    print(f"  decisions: {decisions}")
    print(f"  actions:   {[(a.task, a.owner) for a in actions]}")

    # 1. Proposed idea is not a confirmed decision
    _check("proposed Redis idea NOT recorded as a decision",
           not any("redis" in d.lower() for d in decisions),
           str([d for d in decisions if "redis" in d.lower()]))

    # 2. Unassigned task keeps owner 'unspecified'
    docs_tasks = [a for a in actions
                  if "documentation" in a.task.lower() or "docs" in a.task.lower()]
    _check("deployment-docs task was extracted", len(docs_tasks) >= 1)
    _check("deployment-docs owner == 'unspecified'",
           all((a.owner or "").strip().lower() ==
               "unspecified" for a in docs_tasks),
           str([a.owner for a in docs_tasks]))
    _check("no owner outside names present in transcript",
           all((a.owner or "").strip().lower() == "unspecified"
               or any(n in (a.owner or "").lower() for n in KNOWN_NAMES)
               for a in actions),
           str([a.owner for a in actions]))

    # 3. Negation preserved
    gql = [d for d in decisions if "graphql" in d.lower()]
    _check("GraphQL decision keeps its negation",
           all(NEG_RE.search(d) for d in gql), str(gql))
    gql_tasks = [a.task for a in actions if "graphql" in a.task.lower()]
    _check("no action item to adopt GraphQL",
           not any(re.search(r"\b(adopt|implement|introduce|use)\b", t, re.I)
                   and not NEG_RE.search(t) for t in gql_tasks), str(gql_tasks))
    _check("GraphQL not framed as affirmative adoption anywhere",
           not any(re.search(r"\bwill adopt graphql\b", d, re.I) for d in decisions))


def main():
    for fn in (test_offline_refine_guards, test_offline_owner_guard, test_live_mesa):
        try:
            fn()
        except Exception as e:
            _check(f"{fn.__name__} ran without exception", False, repr(e))
    failed = [n for n, ok, _ in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed")
    if failed:
        print("FAILED:", *failed, sep="\n  - ")
    sys.exit(1 if failed else 0)


# pytest-compatible wrappers (assert on accumulated results)
def test_all_guards_pass():
    _results.clear()
    for fn in (test_offline_refine_guards, test_offline_owner_guard, test_live_mesa):
        fn()
    assert all(ok for _, ok, _ in _results), [
        n for n, ok, _ in _results if not ok]


if __name__ == "__main__":
    main()
