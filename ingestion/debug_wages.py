import re
import pymupdf
import extract_acts as ea

pdf = ea.RAW_DIR / "payment_of_wages_act_1936.pdf"
with pymupdf.open(pdf) as doc:
    total = len(doc)
    print("total pages:", total, "| stop threshold page index:", 0.6 * total)
    for pi, page in enumerate(doc):
        if pi < 18 or pi > 27:
            continue
        in_fn = False
        for raw in page.get_text("text").splitlines():
            line = raw.strip()
            if not line:
                continue
            if pi >= 0.6 * total and (line in ea.STATE_NAMES or ea.STATE_AMENDMENT_START.match(line)):
                print(f"p{pi+1} STOP TRIGGER: {line[:80]}")
                break
            if ea.FOOTNOTE_LINE.match(line) and not in_fn:
                in_fn = True
                print(f"p{pi+1} FOOTNOTE MODE starts at: {line[:80]}")
            if re.match(r"^\d{1,3}[A-Z]{0,2}\.", line):
                print(f"p{pi+1} {'DROPPED' if in_fn else 'kept   '} | {line[:80]}")