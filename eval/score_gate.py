r"""
Score the defer-to-human gate using a saved retrieval run (no models needed).

Usage (from the project root):
    python eval\score_gate.py reranked_small
(reads eval\results_reranked_small.json, which must have been made AFTER the
 score-saving update, i.e. rows contain "top_score").

Metrics:
  correct deferral rate = of questions that SHOULD defer, how many did
  false deferral rate   = of questions that should be ANSWERED, how many wrongly deferred
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from generation.gate import should_defer  # noqa: E402

EVAL_DIR = BASE / "eval"


def evaluate(rows, threshold):
    defer_q = [r for r in rows if r["should_defer"]]
    answer_q = [r for r in rows if not r["should_defer"]]
    d_ok = [r for r in defer_q if should_defer(r["question"], r.get("top_score"), threshold)["defer"]]
    d_bad = [r for r in answer_q if should_defer(r["question"], r.get("top_score"), threshold)["defer"]]
    return {
        "threshold": threshold,
        "correct_deferral": len(d_ok) / len(defer_q) if defer_q else 0.0,
        "false_deferral": len(d_bad) / len(answer_q) if answer_q else 0.0,
        "missed": [r for r in defer_q if r not in d_ok],
        "false": d_bad,
    }


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "reranked_small"
    data = json.load(open(EVAL_DIR / f"results_{label}.json", encoding="utf-8"))
    rows = data["rows"]
    if not any(r.get("top_score") is not None for r in rows):
        print("No top_score in this results file. Re-run score_retrieval.py with the updated "
              "hybrid_search.py first (with 'rerank').")
        return

    n_def = sum(r["should_defer"] for r in rows)
    print(f"{len(rows)} questions: {n_def} should defer, {len(rows) - n_def} should be answered\n")

    rules_only = evaluate(rows, -1e9)
    print("RULES ONLY (confidence layer off):")
    print(f"  correct deferral rate: {rules_only['correct_deferral']:.2f}")
    print(f"  false deferral rate  : {rules_only['false_deferral']:.2f}")
    for r in rules_only["missed"]:
        print(f"  MISSED  {r['id']} {r['question']}  (top_score={r.get('top_score'):.2f})")
    for r in rules_only["false"]:
        print(f"  WRONGLY DEFERRED  {r['id']} {r['question']}")

    scores = sorted({round(r["top_score"], 2) for r in rows if r.get("top_score") is not None})
    print("\nSCORES (rule layer + confidence layer). A question defers if its top score < threshold:")
    print("  threshold | correct deferral | false deferral")
    table = []
    for t in scores:
        res = evaluate(rows, t)
        gain = res["correct_deferral"] - res["false_deferral"]
        table.append((gain, t, res))
        print(f"  {t:9.2f} | {res['correct_deferral']:16.2f} | {res['false_deferral']:14.2f}")

    top_gain = max(x[0] for x in table)
    plateau = [x for x in table if x[0] >= top_gain - 1e-9]
    # Take the middle of the best plateau (by value), not its edge.
    t = (min(x[1] for x in plateau) + max(x[1] for x in plateau)) / 2
    res = evaluate(rows, t)
    no_value = abs(t - min(scores)) < 1e-9 or t <= min(scores) + 0.5
    if no_value:
        print("\nNOTE: the best threshold is just the lowest score, so the confidence layer adds "
              "nothing on this question set (the rules already catch every defer question).")
        print("Keep the confidence layer OFF (do not set LEXIRAG_CONF_THRESHOLD) until you test it "
              "on questions that the rules do NOT catch.")
    print(f"\nBest trade-off on THIS eval set: threshold {t:.2f} "
          f"(correct {res['correct_deferral']:.2f}, false {res['false_deferral']:.2f}).")
    print("Caution: the eval set is small, so treat this as a starting value, not a final one.")
    if not no_value:
        print("To use it:  $env:LEXIRAG_CONF_THRESHOLD = \"%.2f\"" % t)

    print("\nPer-question view at the best threshold:")
    for r in rows:
        d = should_defer(r["question"], r.get("top_score"), t)
        ok = d["defer"] == r["should_defer"]
        print(f"  {'OK ' if ok else 'BAD'} {r['id']} defer={d['defer']!s:5} want={r['should_defer']!s:5} "
              f"reasons={','.join(d['reasons']) or '-'}")


if __name__ == "__main__":
    main()