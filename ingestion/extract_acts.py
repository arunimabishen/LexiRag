"""
Pathway A, step 1: turn real Act PDFs into the same JSON format our pipeline
already uses (id, act, section, title, text).

How it works:
1. Read every page of the PDF with PyMuPDF.
2. Clean out page furniture (gazette headers, page numbers, blank lines).
3. Find section starts using the standard Indian-Act pattern:
       "35. Manner in which complaint shall be made.—(1) A complaint ..."
   i.e. a number (optionally with a letter, like 12A), a dot, a heading that
   ends with ".—" (or ".-" / ". —").
4. The Table of Contents lists every heading too, so each section number
   shows up twice. We keep the occurrence with the LONGEST text (the real one).
5. Very long sections (like Definitions) are split into smaller chunks at
   sub-clause boundaries, because the embedding model only reads ~128 tokens
   well. Chunks keep the same section number so citations stay correct.

Usage (from the project root):
    python ingestion/extract_acts.py
Put PDFs in data/raw/ and register them in ACTS below.
"""

import json
import re
from pathlib import Path

import pymupdf  # PyMuPDF

BASE = Path(__file__).resolve().parent.parent
RAW_DIR = BASE / "data" / "raw"
OUT_FILE = BASE / "data" / "processed" / "sections.json"

# ---------------------------------------------------------------------------
# REGISTER YOUR ACTS HERE. filename must match what you put in data/raw/.
# prefix is used to build chunk ids (CPA-35, MTA-10, ...).
# ---------------------------------------------------------------------------
ACTS = [
    {
        "file": "consumer_protection_act_2019.pdf",
        "act": "Consumer Protection Act, 2019",
        "prefix": "CPA",
        "domain": "consumer",
    },
    {
        "file": "model_tenancy_act_2021.pdf",
        "act": "Model Tenancy Act, 2021",
        "prefix": "MTA",
        "domain": "tenancy",
    },
    {
        "file": "payment_of_wages_act_1936.pdf",
        "act": "Payment of Wages Act, 1936",
        "prefix": "PWA",
        "domain": "labour",
    },
]

MAX_CHUNK_CHARS = 900  # keep chunks small enough for the embedding model

# Real PDFs use many dash characters (em dash, en dash, hyphen, minus, or a
# mangled middle-dot when the font encoding is odd). Accept all of them.
DASHES = "[—–‒―−·-]"

# "35. Manner in which complaint shall be made.—(1) ..."
SECTION_START = re.compile(
    r"^(?P<num>\d{1,3}[A-Z]{0,2})\.\s+(?P<title>[A-Z][^\n]{2,200}?)\.\s*" + DASHES + r"\s*(?P<rest>.*)$"
)
# Heading that sits alone on its line: "35. Manner in which complaint shall be made."
SECTION_HEADING_ONLY = re.compile(
    r"^(?P<num>\d{1,3}[A-Z]{0,2})\.\s+(?P<title>[A-Z][^\n]{2,200}?)\.?$"
)

NOISE_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),  # bare page numbers
    re.compile(r"THE GAZETTE OF INDIA", re.I),
    re.compile(r"^\s*REGISTERED NO", re.I),
    re.compile(r"^\s*(PART|SECTION)\s+[IVX]+", re.I),
    re.compile(r"^\s*\[?(PART II|Part II)", re.I),
    re.compile(r"^\s*www\.", re.I),
    re.compile(r"^\s*(MINISTRY OF LAW AND JUSTICE|Legislative Department)", re.I),
]


def read_pdf_lines(pdf_path):
    lines = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            for line in page.get_text("text").splitlines():
                line = line.strip()
                if not line:
                    continue
                if any(p.search(line) for p in NOISE_PATTERNS):
                    continue
                lines.append(line)
    return lines


def split_into_sections(lines):
    """Walk the lines, starting a new section whenever a heading pattern hits."""
    sections = []
    current = None
    i = 0
    while i < len(lines):
        line = lines[i]
        m = SECTION_START.match(line)
        heading_only = None
        if not m:
            heading_only = SECTION_HEADING_ONLY.match(line)
            # heading on its own line is only valid if the NEXT line continues
            # the body (starts with "(1)" or text), otherwise it's a TOC entry
            # or a stray numbered line.
            if heading_only and i + 1 < len(lines):
                nxt = lines[i + 1]
                if not (nxt.startswith("(") or nxt[:1] in "—–‒―−·-"):
                    heading_only = None
            else:
                heading_only = None

        if m:
            if current:
                sections.append(current)
            current = {
                "num": m.group("num"),
                "title": m.group("title").strip(),
                "text": m.group("rest").strip(),
            }
        elif heading_only:
            if current:
                sections.append(current)
            current = {
                "num": heading_only.group("num"),
                "title": heading_only.group("title").strip().rstrip("."),
                "text": "",
            }
        elif current:
            current["text"] += " " + line
        i += 1

    if current:
        sections.append(current)
    return sections


def dedupe_keep_longest(sections):
    """TOC + body both produce the same section number. Keep the longest text."""
    best = {}
    order = []
    for s in sections:
        key = s["num"]
        s["text"] = re.sub(r"\s+", " ", s["text"]).strip()
        if key not in best:
            best[key] = s
            order.append(key)
        elif len(s["text"]) > len(best[key]["text"]):
            best[key] = s
    return [best[k] for k in order]


def chunk_text(text, max_chars=MAX_CHUNK_CHARS):
    """Split long text at sub-clause markers like (1), (2), (a), (b)."""
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"(?=\s\(\d+\)\s)", text)
    chunks, buf = [], ""
    for p in parts:
        if buf and len(buf) + len(p) > max_chars:
            chunks.append(buf.strip())
            buf = p
        else:
            buf += p
    if buf.strip():
        chunks.append(buf.strip())

    # any chunk STILL too long (no sub-clauses): hard split at sentence ends
    final = []
    for c in chunks:
        if len(c) <= max_chars * 1.5:
            final.append(c)
            continue
        sentences = re.split(r"(?<=[.;:])\s+", c)
        buf = ""
        for s in sentences:
            if buf and len(buf) + len(s) > max_chars:
                final.append(buf.strip())
                buf = s
            else:
                buf += " " + s
        if buf.strip():
            final.append(buf.strip())
    return final


def process_act(cfg):
    pdf_path = RAW_DIR / cfg["file"]
    if not pdf_path.exists():
        print(f"  [skip] {cfg['file']} not found in data/raw/")
        return []

    print(f"  Reading {cfg['file']} ...")
    lines = read_pdf_lines(pdf_path)
    sections = dedupe_keep_longest(split_into_sections(lines))
    print(f"  Found {len(sections)} sections")

    records = []
    for s in sections:
        if len(s["text"]) < 20:
            continue  # empty/TOC-only leftovers
        chunks = chunk_text(s["text"])
        for idx, chunk in enumerate(chunks, start=1):
            suffix = "" if len(chunks) == 1 else f"-p{idx}"
            records.append(
                {
                    "id": f"{cfg['prefix']}-{s['num']}{suffix}",
                    "act": cfg["act"],
                    "section": f"Section {s['num']}",
                    "title": s["title"],
                    "domain": cfg["domain"],
                    "text": chunk,
                }
            )
    return records


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_records = []
    for cfg in ACTS:
        print(f"\n{cfg['act']}")
        all_records.extend(process_act(cfg))

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(all_records)} chunks to {OUT_FILE}")
    if all_records:
        print("\nSanity check, first 5 chunks:")
        for r in all_records[:5]:
            print(f"  {r['id']:12} | {r['section']:12} | {r['title'][:50]} | {len(r['text'])} chars")


if __name__ == "__main__":
    main()
