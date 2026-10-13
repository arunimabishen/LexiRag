import pymupdf
from pathlib import Path

pdf = Path(__file__).resolve().parent.parent / "data" / "raw" / "consumer_protection_act_2019.pdf"

with pymupdf.open(pdf) as doc:
    for page_no in [0, 1, 9, 10]:
        if page_no >= len(doc):
            continue

        page = doc[page_no]
        print(f"\n{'=' * 25} PAGE {page_no + 1} {'=' * 25}")

        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue

            x0, y0, x1, y1 = block["bbox"]
            text = " ".join(
                " ".join(span["text"] for span in line["spans"])
                for line in block["lines"]
            ).strip()

            if text:
                area = (
                    "LEFT MARGIN" if x1 <= 112
                    else "RIGHT MARGIN" if x0 >= 482
                    else "BODY"
                )
                print(
                    f"{area:12} "
                    f"x0={x0:6.1f} y0={y0:6.1f} "
                    f"x1={x1:6.1f} y1={y1:6.1f} | {text[:180]}"
                )
