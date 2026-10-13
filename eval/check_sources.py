r"""
Quick look at what retrieval returns for a few everyday questions (no answer generation).
Shows the top 8 chunks so you can see whether the key section would appear in the top 3 or top 5.

Usage (from the project root):
    python eval\check_sources.py
    python eval\check_sources.py "your own question"
"""
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass
os.environ.setdefault("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
os.environ.setdefault("RERANK_POOL", "20")

from generation.language import detect_language  # noqa: E402
from retrieval.hybrid_search import hybrid_search  # noqa: E402
from retrieval.query_rewrite import rewrite_query  # noqa: E402

DEFAULT = [
    "मेरी सैलरी नहीं मिली",
    "meri salary nahi mili",
    "My employer has not paid my salary for two months. What can I do?",
    "My landlord is not returning my security deposit",
    "I bought a defective phone and the shop refuses to refund me",
]

for q in (sys.argv[1:] or DEFAULT):
    lang = detect_language(q)
    search_q = q
    if lang != "en":
        search_q = rewrite_query(q) or q
    print("=" * 78)
    print(f"{q}   [{lang}]   searched as: {search_q}")
    for i, s in enumerate(hybrid_search(search_q, top_k=8, rerank=True), start=1):
        mark = "  <- top 3" if i <= 3 else ("  <- top 5" if i <= 5 else "")
        print(f"  {i}. {s['id']:<16} {s['section']:<14} {s['title'][:50]}{mark}")