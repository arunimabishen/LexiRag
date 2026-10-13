import sys
from pathlib import Path
import pymupdf

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
name, pno = sys.argv[1], int(sys.argv[2])
with pymupdf.open(RAW / name) as doc:
    page = doc[pno - 1]
    print("page width", page.rect.width)
    for b in sorted(page.get_text("blocks"), key=lambda b: (b[1], b[0])):
        print(f"x0={b[0]:.0f} x1={b[2]:.0f} y0={b[1]:.0f} | {b[4].strip()[:70]!r}")