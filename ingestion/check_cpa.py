import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "processed" / "sections.json"
recs = json.load(open(DATA, encoding="utf-8"))
cpa = [r for r in recs if r["id"].startswith("CPA-")]
print(f"{len(cpa)} CPA chunks\n")

print("== Spot check (first chunk of each) ==")
want = ["Section 1", "Section 2(1)", "Section 2(7)", "Section 2(47)",
        "Section 35", "Section 36", "Section 37", "Section 39"]
for w in want:
    r = next((x for x in cpa if x["section"] == w), None)
    if r:
        print(f"{r['id']} | {r['title']} | {r['text'][:90]}")
    else:
        print(f"{w}: NOT FOUND")

print("\n== Chunks whose title looks like a fallback (starts with '(' or ends with '...') ==")
bad = {}
for r in cpa:
    if r["title"].endswith("...") or r["title"].startswith("("):
        bad.setdefault(r["section"], r["title"])
for sec, t in list(bad.items())[:20]:
    print(f"{sec} | {t}")
print("none" if not bad else f"{len(bad)} sections")

print("\n== Another section's title appearing inside a chunk's text ==")
titles = {r["title"] for r in cpa if not r["title"].startswith("Definition")}
hits = 0
for r in cpa:
    low = r["text"].lower()
    for t in titles:
        if t != r["title"] and len(t) > 15 and t.lower() in low:
            print(f"{r['id']} contains {t!r}")
            hits += 1
print("none" if hits == 0 else f"{hits} hits (some may be legitimate cross-references)")

print("\n== Chapter banners left in text ==")
n = 0
for r in cpa:
    if re.search(r"\bCHAPTER [IVXLC]+\b", r["text"]):
        print(r["id"])
        n += 1
print("none" if n == 0 else f"{n} chunks")