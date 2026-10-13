import re
import sys
from pathlib import Path
import pymupdf

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
pdf = sys.argv[1]
nums = set(sys.argv[2:])
pat = re.compile(r"^\D{0,4}(\d{1,3}[A-Z]{0,2})\s*[\.\[]")

with pymupdf.open(RAW / pdf) as doc:
    for pno, page in enumerate(doc, start=1):
        for line in page.get_text("text").splitlines():
            s = line.strip()
            m = pat.match(s)
            if m and m.group(1) in nums:
                print(f"p{pno} | {s[:110]}")