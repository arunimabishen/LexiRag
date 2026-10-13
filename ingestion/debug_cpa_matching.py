import re
import pymupdf
from pathlib import Path

pdf = Path("data/raw/consumer_protection_act_2019.pdf")

num_line = re.compile(
    r"^(?P<num>\d{1,3}[A-Z]{0,2})\.{1,2}\s*(?P<rest>\S.*)$"
)

with pymupdf.open(pdf) as doc:
    for pno in [0, 1, 9, 10]:
        page = doc[pno]
        print(f"\n--- PAGE {pno + 1} ---")

        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue

            x0, y0, x1, y1 = block["bbox"]

            for line in block["lines"]:
                text = " ".join(
                    "".join(span["text"] for span in line["spans"]).split()
                )
                if not text:
                    continue

                ly, lx = line["bbox"][1], line["bbox"][0]
                m = num_line.match(text)

                if x1 <= 112 or x0 >= 482:
                    print(f"NOTE y={y0:.1f}: {text[:100]}")
                elif ly >= 85 and lx < 125 and m:
                    print(
                        f"CANDIDATE y={ly:.1f}, num={m.group('num')}, "
                        f"rest={m.group('rest')[:65]!r}"
                    )
