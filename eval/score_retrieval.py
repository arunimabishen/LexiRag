r"""
Score retrieval against eval/eval_set.json.

Usage (from the project root):
    python eval\score_retrieval.py baseline            (no cross-encoder)
    python eval\score_retrieval.py reranked rerank     (with cross-encoder)
    python eval\score_retrieval.py holdout rerank eval\gate_holdout.json   (another question file)
    python eval\score_retrieval.py rewrite rerank rewrite     (rewrite Hindi/Hinglish questions to English)
    python eval\score_retrieval.py route rerank rewrite route (also route to the right Act(s))
The label is just a name for this run; results are saved to eval\results_<label>.json
so you can compare runs later (e.g. before / after the re-ranker).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
try:  # print Hindi correctly even when output is piped (Windows would use cp1252)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from retrieval.domain_router import route_domains  # noqa: E402
from retrieval.hybrid_search import routed_search  # noqa: E402
from retrieval.query_rewrite import rewrite_query  # noqa: E402

EVAL_FILE = BASE / "eval" / "eval_set.json"
SECTIONS_FILE = BASE / "data" / "processed" / "sections.json"
KS = (1, 3, 5)


def key_of(rec):
    """'CPA|Section 35' from a chunk record (id prefix + section label)."""
    return f"{rec['id'].split('-')[0]}|{rec['section']}"


def satisfied(gold, found, mode):
    if not gold:
        return None
    hits = [g in found for g in gold]
    return all(hits) if mode == "all" else any(hits)


def main():
    args = sys.argv[1:]
    label = args[0] if args else "run"
    flags = args[1:]
    use_rerank = "rerank" in flags
    use_rewrite = "rewrite" in flags
    use_route = "route" in flags
    files = [a for a in flags if a.endswith(".json")]
    eval_file = Path(files[0]) if files else EVAL_FILE
    if not eval_file.is_absolute():
        eval_file = BASE / eval_file
    print(f"Cross-encoder re-ranking: {'ON' if use_rerank else 'off'}")
    print(f"Hindi/Hinglish query rewriting: {'ON' if use_rewrite else 'off'}")
    print(f"Domain routing: {'ON' if use_route else 'off'}")

    questions = json.load(open(eval_file, encoding="utf-8"))
    sections = json.load(open(SECTIONS_FILE, encoding="utf-8"))
    known = {key_of(r) for r in sections}

    bad = sorted({g for q in questions for g in q["gold"] if g not in known})
    if bad:
        print("WARNING: gold sections not found in sections.json (fix data or eval set):")
        for b in bad:
            print("   ", b)
        print()

    rows = []
    for q in questions:
        search_text = q["question"]
        if use_rewrite and q["lang"] != "en":
            search_text = rewrite_query(q["question"]) or search_text
        domains = route_domains(search_text) if use_route else None
        results = routed_search(search_text, top_k=max(KS), rerank=use_rerank, domains=domains)
        print(f"  {q['id']} done ({len(rows) + 1}/{len(questions)})", flush=True)
        ranked = [key_of(r) for r in results]
        row = {"id": q["id"], "category": q["category"], "lang": q["lang"],
               "question": q["question"], "gold": q["gold"], "mode": q["mode"],
               "should_defer": q["should_defer"], "retrieved": ranked,
               "top_score": results[0].get("_score") if results else None,
               "search_query": search_text, "domains": domains}
        if q["gold"]:
            for k in KS:
                row[f"hit@{k}"] = satisfied(q["gold"], set(ranked[:k]), q["mode"])
            first = next((i + 1 for i, r in enumerate(ranked) if r in q["gold"]), None)
            row["rr"] = 1.0 / first if first else 0.0
        rows.append(row)

    scored = [r for r in rows if r["gold"]]
    if not scored:
        out = BASE / "eval" / f"results_{label}.json"
        json.dump({"label": label, "overall": {}, "rows": rows},
                  open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\nNo questions with gold sections in this file, so no retrieval scores. Saved {out}")
        return

    def summarize(subset):
        n = len(subset)
        out = {f"hit@{k}": sum(1 for r in subset if r[f"hit@{k}"]) / n for k in KS}
        out["mrr"] = sum(r["rr"] for r in subset) / n
        out["n"] = n
        return out

    overall = summarize(scored)

    # Save first, so a printing problem can never lose a finished run.
    out = BASE / "eval" / f"results_{label}.json"
    json.dump({"label": label, "overall": overall, "rows": rows},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    by_cat, by_lang = defaultdict(list), defaultdict(list)
    for r in scored:
        by_cat[r["category"]].append(r)
        by_lang[r["lang"]].append(r)

    def line(name, s):
        return (f"{name:16} n={s['n']:<3} hit@1={s['hit@1']:.2f}  hit@3={s['hit@3']:.2f}  "
                f"hit@5={s['hit@5']:.2f}  MRR={s['mrr']:.2f}")

    print("=" * 78)
    print(f"RETRIEVAL RESULTS ({label})  -  {len(scored)} scored questions "
          f"(+{len(rows) - len(scored)} defer questions not scored here)")
    print("=" * 78)
    print(line("OVERALL", overall))
    print("\nBy category:")
    for c, rs in sorted(by_cat.items()):
        print("  " + line(c, summarize(rs)))
    print("\nBy language:")
    for c, rs in sorted(by_lang.items()):
        print("  " + line(c, summarize(rs)))

    fails = [r for r in scored if not r["hit@3"]]
    print(f"\nFAILURES at top-3: {len(fails)}")
    for r in fails:
        print(f"\n  {r['id']} [{r['category']}] {r['question']}")
        print(f"     wanted   : {' | '.join(r['gold'])}  ({r['mode']})")
        print(f"     retrieved: {' | '.join(r['retrieved'][:3])}")

    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()