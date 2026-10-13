import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "processed" / "sections.json"
recs = json.load(open(DATA, encoding="utf-8"))
seen = set()
for r in recs:
    if r["id"].startswith("CPA-2-cl"):
        continue  # skip the 47 definition clauses
    key = (r["act"], r["section"])
    if key in seen:
        continue
    seen.add(key)
    print(f"{r['id']:14} | {r['section']:14} | {r['title'][:60]}")