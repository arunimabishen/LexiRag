"""
Debug helper: find where specific section numbers appear in a PDF, using the
RAW lines (nothing filtered), so we can see why the splitter missed them.

Usage (from the project root):
    python ingestion/find_numbers.py payment_of_wages_act_1936.pdf 9 24
"""

import re
import sys
from pathlib import Path

import pymupdf

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main():
    if len(sys.argv) < 2:
        print("usage: python ingestion/find_numbers.py FILE.pdf [number ...]")
        return
    path = RAW_DIR / sys.argv[1]
    nums = sys.argv[2:]
    if not path.exists():
        print("not found:", path)
        return

    lines = []
    with pymupdf.open(path) as doc:
        for pno, page in enumerate(doc, start=1):
            for l in page.get_text("text").splitlines():
                if l.strip():
                    lines.append((pno, l.strip()))

    print(f"{sys.argv[1]}: {len(lines)} non-empty lines")

    for n in nums:
        print("=" * 60)
        print(f"Looking for section {n}")
        start = re.compile(r"^[\[\(\s]*" + re.escape(n) + r"[A-Z]?\s*[\.\)\]]")
        anywhere = re.compile(r"(?<!\d)" + re.escape(n) + r"\s*\.\s+[A-Z(]")
        hits = [i for i, (_, l) in enumerate(lines) if start.search(l)]
        if hits:
            for i in hits[:4]:
                print(f"  [line-start match, page {lines[i][0]}]")
                for j in range(max(0, i - 1), min(len(lines), i + 3)):
                    print("    |", lines[j][1][:110])
        else:
            print("  no line starts with this number")
            any_hits = [i for i, (_, l) in enumerate(lines) if anywhere.search(l)]
            for i in any_hits[:3]:
                print(f"  [appears mid-line, page {lines[i][0]}]")
                print("    |", lines[i][1][:140])

    print("=" * 60)
    print("Last 12 lines of the document:")
    for pno, l in lines[-12:]:
        print(f"  | p{pno}: {l[:110]}")


if __name__ == "__main__":
    main()