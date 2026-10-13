r"""
Print one comparison table of saved retrieval runs.   python eval\compare_runs.py
Reads eval\results_<label>.json for each label below (skips any that do not exist).
"""
import json
from pathlib import Path

EVAL = Path(__file__).resolve().parent
RUNS = [
    ("reranked_small",   "MAIN  before fixes (old data)"),
    ("clean_rerank",     "MAIN  cleaned data"),
    ("rewrite",          "MAIN  cleaned data + query rewrite"),
    ("route",            "MAIN  rewrite + domain routing"),
    ("nohack",           "MAIN  rewrite, deposit bonus OFF (current default)"),
    ("rewrite_v2",       "MAIN  rewrite v2 (adds Act vocabulary)"),
    ("holdout",          "HOLDOUT before fixes (old data)"),
    ("holdout_clean",    "HOLDOUT cleaned data"),
    ("holdout_rewrite",  "HOLDOUT cleaned data + query rewrite"),
    ("holdout_route",    "HOLDOUT rewrite + domain routing"),
    ("holdout_rewrite_v2", "HOLDOUT rewrite v2 (adds Act vocabulary)"),
]


def metrics(rows):
    scored = [r for r in rows if r.get("gold")]
    if not scored:
        return None
    n = len(scored)
    return (n, sum(1 for r in scored if r["hit@1"]) / n, sum(1 for r in scored if r["hit@3"]) / n,
            sum(1 for r in scored if r["hit@5"]) / n, sum(r["rr"] for r in scored) / n)


def line(name, m):
    return f"{name:40} n={m[0]:<3} hit@1={m[1]:.2f}  hit@3={m[2]:.2f}  hit@5={m[3]:.2f}  MRR={m[4]:.2f}"


print("=" * 100)
for label, desc in RUNS:
    path = EVAL / f"results_{label}.json"
    if not path.exists():
        continue
    rows = json.load(open(path, encoding="utf-8"))["rows"]
    m = metrics(rows)
    if m:
        print(line(desc, m))
        non_en = [r for r in rows if r.get("lang") in ("hi", "hinglish")]
        mm = metrics(non_en)
        if mm:
            print(line("      Hindi + Hinglish only", mm))
        multi = [r for r in rows if r.get("category") == "multi_domain"]
        mx = metrics(multi)
        if mx:
            print(line("      Multi-domain only", mx))
print("=" * 100)