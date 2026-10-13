r"""
Score the LLM scope check on BOTH question files (main set + holdout).

Usage (from the project root, GROQ_API_KEY must be set):
    python eval\score_scope.py

For each question it asks the scope checker, then combines that verdict with the
keyword rules (the same way the gate will) and reports:
  correct deferral rate = of questions that SHOULD defer, how many did
  false deferral rate   = of questions that should be ANSWERED, how many were refused
Answers are cached in eval\scope_cache.json so re-runs do not repeat API calls.
"""
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from generation.gate import should_defer  # noqa: E402
from generation.scope_check import GROQ_API_KEY, check_scope  # noqa: E402

FILES = [("main", BASE / "eval" / "eval_set.json"),
         ("holdout", BASE / "eval" / "gate_holdout.json")]
CACHE = BASE / "eval" / "scope_cache.json"


def main():
    if not GROQ_API_KEY:
        print("GROQ_API_KEY is not set in this terminal. Set it first, then run again.")
        return
    cache = json.load(open(CACHE, encoding="utf-8")) if CACHE.exists() else {}
    summary = {}

    for name, path in FILES:
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        questions = json.load(open(path, encoding="utf-8"))
        rows, errors = [], 0
        for i, q in enumerate(questions, start=1):
            text = q["question"]
            if text not in cache:
                cache[text] = check_scope(text)
                json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                time.sleep(2.2)  # stay under the free-tier request limit
            scope = cache[text]
            if scope is None:
                errors += 1
            decision = should_defer(text, scope=scope)
            rows.append((q, scope, decision))
            print(f"  {name} {q['id']} done ({i}/{len(questions)})", flush=True)

        defer_q = [r for r in rows if r[0]["should_defer"]]
        answer_q = [r for r in rows if not r[0]["should_defer"]]
        caught = [r for r in defer_q if r[2]["defer"]]
        wrong = [r for r in answer_q if r[2]["defer"]]
        summary[name] = (len(caught), len(defer_q), len(wrong), len(answer_q), errors)

        print("\n" + "=" * 70)
        print(f"{name.upper()}: {len(questions)} questions ({errors} scope-check errors)")
        print("=" * 70)
        if defer_q:
            print(f"correct deferral rate: {len(caught)}/{len(defer_q)} = {len(caught)/len(defer_q):.2f}")
        if answer_q:
            print(f"false deferral rate  : {len(wrong)}/{len(answer_q)} = {len(wrong)/len(answer_q):.2f}")
        for q, scope, d in defer_q:
            if not d["defer"]:
                print(f"  MISSED   {q['id']} {q['question']}\n           scope={scope}")
        for q, scope, d in answer_q:
            if d["defer"]:
                print(f"  REFUSED  {q['id']} {q['question']}\n           reasons={d['reasons']} scope={scope}")
        print()

    print("SUMMARY")
    for name, (c, nd, w, na, er) in summary.items():
        print(f"  {name:8} correct deferral {c}/{nd}   false deferral {w}/{na}   errors {er}")
    print("\nCaution: small sets. The holdout questions were written by the assistant, "
          "so call them a stress test in the paper.")


if __name__ == "__main__":
    main()