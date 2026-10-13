import sys
from pathlib import Path

import pymupdf

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DEFAULT_FILES = [
    "consumer_protection_act_2019.pdf",
    "payment_of_wages_act_1936.pdf",
    "chhattisgarh_rent_control_act_2011.pdf",
]


def peek(name):
    path = RAW_DIR / name
    print("=" * 70)
    print(name)
    if not path.exists():
        print("  not found in data/raw/")
        return
    with pymupdf.open(path) as doc:
        n = len(doc)
        page_no = min(n - 1, max(0, n // 3))
        page = doc[page_no]
        raw = page.get_text("text")
        blocks = page.get_text("blocks")
        width = page.rect.width
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    print(f"pages={n}, page {page_no + 1}, page width={width:.0f}")
    print("-- text mode, first 30 lines:")
    for l in lines[:30]:
        print("  |", l[:110])
    print("-- blocks mode (x0, y0 | text), first 15:")
    for b in blocks[:15]:
        print(f"  | x={b[0]:.0f} y={b[1]:.0f} | {b[4].strip()[:90]!r}")


if __name__ == "__main__":
    print("peek_pdf started")
    targets = sys.argv[1:] or DEFAULT_FILES
    for t in targets:
        peek(t)