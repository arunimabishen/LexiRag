"""
Pathway A, step 1: turn real Act PDFs into the JSON format our pipeline uses
(id, act, section, title, domain, text).

Usage from the project root:
    python ingestion/extract_acts.py
"""

import json
import re
from pathlib import Path

import pymupdf  # PyMuPDF


BASE = Path(__file__).resolve().parent.parent
RAW_DIR = BASE / "data" / "raw"
OUT_FILE = BASE / "data" / "processed" / "sections.json"


ACTS = [
    {
        "file": "consumer_protection_act_2019.pdf",
        "act": "Consumer Protection Act, 2019",
        "prefix": "CPA",
        "domain": "consumer",
        "margin": True,
        "definitions": "2",
    },
    {
        "file": "payment_of_wages_act_1936.pdf",
        "act": "Payment of Wages Act, 1936",
        "prefix": "PWA",
        "domain": "labour",
    },
    {
        "file": "chhattisgarh_rent_control_act_2011.pdf",
        "act": "Chhattisgarh Rent Control Act, 2011",
        "prefix": "CGRCA",
        "domain": "tenancy",
        "schedules": True,
    },
    {
        "file": "legal_services_authorities_act_1987.pdf",
        "act": "Legal Services Authorities Act, 1987",
        "prefix": "LSAA",
        "domain": "legal_aid",
    },
    {
        "file": "nalsa_free_competent_legal_services_regs_2010.pdf",
        "act": "NALSA (Free and Competent Legal Services) Regulations, 2010",
        "prefix": "NALSA-FCLS",
        "domain": "legal_aid",
        "label": "Regulation",
        "ocr": True,
        "skip": True,
    },
    {
        "file": "nalsa_legal_services_clinics_regs_2011.pdf",
        "act": "NALSA (Legal Services Clinics) Regulations, 2011",
        "prefix": "NALSA-LSC",
        "domain": "legal_aid",
        "label": "Regulation",
        "ocr": True,
        "skip": True,
    },
]


MAX_CHUNK_CHARS = 900
MAX_GAP = 10

DASHES = "[—–‒―−·_-]"

NUM_LINE = re.compile(
    r"^(?P<num>\d{1,3}[A-Z]{0,2})\.{1,2}\s*(?P<rest>\S.*)$"
)

OCR_NUM_LINE = re.compile(
    r"^(?P<num>[0-9lIOo|]{1,3}[A-Z]{0,2})\.{1,2}\s*(?P<rest>\S.*)$"
)

OCR_FIX = str.maketrans("lIOo|", "11001")

SCHEDULE_LINE = re.compile(r"^Schedule\s+(\d+)$", re.I)

INLINE_TITLE = re.compile(
    r"^(?P<title>[A-Z][^\n]{2,120}?)\.\s*" + DASHES + r"+\s*(?P<body>.*)$"
)

HEADING_ONLY = re.compile(r"^[A-Z][^()\n]{2,150}?\.$")


NOISE_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"THE GAZETTE OF INDIA", re.I),
    re.compile(r"^\s*REGISTERED NO", re.I),
    re.compile(r"^\s*(PART|SECTION)\s+[IVX]+", re.I),
    re.compile(r"^\s*\[?(PART II|Part II)", re.I),
    re.compile(r"^\s*SEC\.\s*\d+\s*\]", re.I),
    re.compile(r"^\s*www\.", re.I),
    re.compile(
        r"^\s*(MINISTRY OF LAW AND JUSTICE|Legislative Department)",
        re.I,
    ),
    re.compile(r"Indian Kanoon", re.I),
]


FOOTNOTE_ACTS = {"Payment of Wages Act, 1936"}

FOOTNOTE_LINE = re.compile(
    r"^\d{1,3}\.\s+(?:"
    r"(?:Ins\.|Subs\.|Rep\.|Reg\.)"
    r"|(?:Inserted|Substituted|Omitted|Added|Renumbered|Re-numbered|Repealed|"
    r"Clauses?|Sub-sections?|Sub-clauses?|Sections?|See|Vide|Proviso|"
    r"The\s+words?|The\s+figures?|The\s+brackets?)\b"
    r"|\d{1,2}(?:st|nd|rd|th)\s+[A-Z][a-z]+\s+\d{4}"
    r")"
)
AMENDMENT_MARK = re.compile(r"(?<![\w.])\d{1,2}\[")
STATE_AMENDMENT_START = re.compile(r"^Amendment of section\s+\d+", re.I)
STATE_NAMES = {
    "Andhra Pradesh", "Assam", "Bihar", "Chhattisgarh", "Delhi", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Odisha", "Orissa",
    "Punjab", "Rajasthan", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal",
}


# Acts where India Code prints amendment marks like "3[6. Constitution..."
MARK_ACTS = {"Legal Services Authorities Act, 1987"}

# A real section heading line: "16. Single application in respect of claims.—(1) ..."
BODY_HEAD = re.compile(
    r"^(?P<n>\d{1,3})[A-Z]{0,2}\.\s+(?P<title>[A-Z][^\n]{2,200}?)\.\s*" + DASHES
)
STATE_HEADER = re.compile(r"^STATE AMENDMENTS?\.?$", re.I)
AMENDING_TITLE = re.compile(
    r"^(Amendment|Insertion|Substitution|Short title|Repeal|Omission)", re.I
)


def read_pdf_lines(pdf_path, act_name=""):
    lines = []
    clean_footnotes = act_name in FOOTNOTE_ACTS
    strip_marks = act_name in MARK_ACTS
    with pymupdf.open(pdf_path) as doc:
        total_pages = len(doc)
        last_num = 0          # last real section number seen in the body
        state_mode = False    # True while inside an inline state-amendment block
        for page_index, page in enumerate(doc):
            in_footnotes = False
            for line in page.get_text("text").splitlines():
                line = line.strip()
                if not line:
                    continue
                if strip_marks:
                    line = AMENDMENT_MARK.sub("", line).strip()
                    if not line:
                        continue
                if clean_footnotes:
                    probe = AMENDMENT_MARK.sub("", line).strip()
                    # a state-amendment block starts: skip until the next real section
                    if (
                        line in STATE_NAMES
                        or STATE_HEADER.match(line)
                        or STATE_AMENDMENT_START.match(line)
                    ):
                        state_mode = True
                        continue
                    if state_mode:
                        hm = BODY_HEAD.match(probe)
                        if (
                            hm
                            and last_num < int(hm.group("n")) <= last_num + MAX_GAP
                            and not AMENDING_TITLE.match(hm.group("title"))
                        ):
                            state_mode = False
                            in_footnotes = False
                        else:
                            continue
                    # once a footnote starts, the rest of that page is footnotes
                    if FOOTNOTE_LINE.match(line):
                        in_footnotes = True
                    if in_footnotes:
                        continue
                    line = probe
                    if not line:
                        continue
                    hm = BODY_HEAD.match(line)
                    if hm:  # table-of-contents lines have no ".—" so never match
                        n = int(hm.group("n"))
                        if last_num == 0 or last_num < n <= last_num + MAX_GAP:
                            last_num = n
                if act_name and line.lower() == act_name.lower():
                    continue
                if any(pattern.search(line) for pattern in NOISE_PATTERNS):
                    continue
                lines.append(line)
    return lines


# ---------------------------------------------------------
# Positional PDF extraction for Acts with marginal titles
# ---------------------------------------------------------

MARGIN_LEFT_X1 = 112
MARGIN_RIGHT_X0 = 482
HEADER_Y = 85
BODY_X_MAX = 125
Y_TOL = 10

XREF_NOTE = re.compile(r"^(?:\d+\s+of\s+\d{4}\.?\s*)+$")
CHAPTER_LINE = re.compile(r"^CHAPTER\s+[IVXLC]+\.?$")


def is_margin_junk(text):
    """Identify margin text that is not a section title."""
    return bool(
        XREF_NOTE.match(text)
        or "Digitally signed" in text
        or text.isupper()
    )


def is_body_junk(text, act_name=""):
    """Identify headers, chapter banners, and other unwanted PDF text."""
    if act_name and text.lower() == act_name.lower():
        return True

    if CHAPTER_LINE.match(text):
        return True

    if text.isupper() and len(text) > 8:
        return True

    return any(pattern.search(text) for pattern in NOISE_PATTERNS)


def read_pdf_positional(pdf_path, act_name=""):
    """
    Return:
        body_lines, margin_titles, warnings

    Extract body text and marginal notes separately. Match each marginal
    title to a section heading using its vertical position on the page.
    """

    lines = []
    margin_titles = {}
    warnings = []

    with pymupdf.open(pdf_path) as doc:
        for pno, page in enumerate(doc, start=1):
            body_blocks = []
            notes = []

            page_dict = page.get_text("dict")

            for block in page_dict["blocks"]:
                if block.get("type") != 0:
                    continue

                bx0, by0, bx1, by1 = block["bbox"]
                block_lines = []

                for line in block["lines"]:
                    text = " ".join(
                        "".join(span["text"] for span in line["spans"]).split()
                    )

                    if text:
                        block_lines.append(
                            (line["bbox"][1], line["bbox"][0], text)
                        )

                if not block_lines:
                    continue

                # Identify text blocks located in either margin.
                if bx1 <= MARGIN_LEFT_X1 or bx0 >= MARGIN_RIGHT_X0:
                    note = " ".join(
                        text for _, _, text in block_lines
                    )

                    if by0 >= HEADER_Y and not is_margin_junk(note):
                        notes.append((by0, note))

                else:
                    body_blocks.append((by0, bx0, block_lines))

            body_blocks.sort(key=lambda block: (block[0], block[1]))

            # Find section headings line by line.
            # IMPORTANT: Do not reject headings based on their x-coordinate.
            # Some valid section headings start farther right on the page.
            candidates = []

            for _, _, block_lines in body_blocks:
                for line_y, line_x, text in block_lines:
                    if line_y < HEADER_Y:
                        continue

                    match = NUM_LINE.match(text)

                    if not match:
                        continue

                    rest = match.group("rest")

                    if not rest:
                        continue

                    if (
                        rest[0] in "(“\"'"
                        or rest[0].isupper()
                    ):
                        candidates.append(
                            (line_y, match.group("num"))
                        )

            # Match each margin note to the closest section heading.
            possible_pairs = sorted(
                (
                    abs(note_y - candidate_y),
                    note_index,
                    candidate_index,
                )
                for note_index, (note_y, _) in enumerate(notes)
                for candidate_index, (candidate_y, _) in enumerate(candidates)
                if abs(note_y - candidate_y) <= Y_TOL
            )

            used_notes = set()
            used_candidates = set()

            for _, note_index, candidate_index in possible_pairs:
                if note_index in used_notes:
                    continue

                if candidate_index in used_candidates:
                    continue

                used_notes.add(note_index)
                used_candidates.add(candidate_index)

                title = notes[note_index][1].rstrip(". ").strip()
                section_number = candidates[candidate_index][1]

                margin_titles.setdefault(section_number, title)

            # Report margin notes that could not be matched.
            for note_index, (note_y, note_text) in enumerate(notes):
                if note_index not in used_notes:
                    warnings.append(
                        f"p{pno}: unmatched margin note "
                        f"y={note_y:.0f} {note_text!r}"
                    )

            # Keep cleaned body text, excluding headers and chapter banners.
            for _, _, block_lines in body_blocks:
                for line_y, line_x, text in block_lines:
                    if line_y < HEADER_Y:
                        continue

                    if is_body_junk(text, act_name):
                        continue

                    lines.append(text)

    return lines, margin_titles, warnings


# ---------------------------------------------------------
# Section parsing
# ---------------------------------------------------------

def parse_num(num):
    match = re.match(r"^(\d+)([A-Z]*)$", num)

    return int(match.group(1)), match.group(2)


def split_into_sections(lines, schedules=False, ocr=False):
    sections = []
    titles = {}

    current = None

    last_n = 0
    last_suf = ""

    restarts_left = 1
    in_schedule = False

    def toc_like():
        """Determine whether the collected text resembles a table of contents."""
        lengths = [len(section["text"]) for section in sections]

        if current:
            lengths.append(len(current["text"]))

        return bool(lengths) and sum(lengths) / len(lengths) < 150

    def accept(number, suffix, strict=False):
        nonlocal last_n, last_suf, restarts_left

        accepted = False

        if strict:
            accepted = number == last_n + 1 and not suffix

        elif number == last_n and suffix and suffix > last_suf:
            accepted = True

        elif number > last_n and number - last_n <= MAX_GAP:
            accepted = True

        elif (
            number == 1
            and not suffix
            and last_n > 5
            and restarts_left > 0
            and toc_like()
        ):
            restarts_left -= 1
            accepted = True

        if accepted:
            last_n = number
            last_suf = suffix

        return accepted

    for line in lines:

        # Handle schedules separately.
        if schedules:
            schedule_match = SCHEDULE_LINE.match(line)

            if schedule_match:
                key = "SCH" + schedule_match.group(1)

                if not (current and current["num"] == key):
                    if current:
                        sections.append(current)

                    current = {
                        "num": key,
                        "inline_title": f"Schedule {schedule_match.group(1)}",
                        "text": "",
                    }

                in_schedule = True
                continue

        match = None if in_schedule else (
            OCR_NUM_LINE if ocr else NUM_LINE
        ).match(line)

        if match:
            raw_number = match.group("num")

            if ocr:
                core_match = re.match(r"[0-9lIOo|]+", raw_number)
            else:
                core_match = re.match(r"\d+", raw_number)

            core = core_match.group(0)

            fixed_core = core.translate(OCR_FIX) if ocr else core
            number_text = fixed_core + raw_number[len(core):]

            strict = fixed_core != core

            number, suffix = parse_num(number_text)

            rest = match.group("rest").strip()

            rest = re.sub(
                r"^[—–‒―−·_.\-]+\s*",
                "",
                rest,
            )

            bracket_prefixed = rest.startswith("[")
            rest = re.sub(r"^\[\s*", "", rest)

            if not rest:
                rest = "?"

            starts_correctly = (
                rest[0] in "(“\"'"
                or rest[0].isupper()
            )

            if starts_correctly and accept(number, suffix, strict):
                if current:
                    sections.append(current)

                current = {
                    "num": number_text,
                    "inline_title": None,
                    "text": rest,
                }

                inline_match = INLINE_TITLE.match(rest)

                if inline_match:
                    current["inline_title"] = (
                        inline_match.group("title").strip()
                    )

                    current["text"] = inline_match.group("body").strip()

                elif HEADING_ONLY.match(rest) and len(rest) <= 150:
                    heading = rest.rstrip(".").strip()

                    titles.setdefault(number_text, heading)
                    current["text"] = ""

                elif bracket_prefixed:
                    bm = re.match(
                        r"^(?P<t>[A-Z][^\[\]()]{2,120}?)\.\s*(?P<b>.*)$", rest
                    )
                    if bm:
                        titles.setdefault(number_text, bm.group("t").strip())
                        current["text"] = bm.group("b").strip()

                continue

        if current:
            current["text"] += " " + line

    if current:
        sections.append(current)

    return sections, titles


def dedupe_keep_longest(sections):
    """Keep the longest version of each section."""
    best = {}
    order = []

    for section in sections:
        key = section["num"]

        section["text"] = re.sub(
            r"\s+",
            " ",
            section["text"],
        ).strip()

        if key not in best:
            best[key] = section
            order.append(key)

        elif len(section["text"]) > len(best[key]["text"]):
            if (
                not section.get("inline_title")
                and best[key].get("inline_title")
            ):
                section["inline_title"] = best[key]["inline_title"]

            best[key] = section

    return [best[key] for key in order]


SCHEDULE_TITLES = {
    "SCH1": "Rights of the tenant (Schedule 1, see Section 12(1))",
    "SCH2": "Rights of the landlord (Schedule 2, see Section 12(2))",
    "SCH3": "Obligations of the landlord (Schedule 3, see Section 12(3))",
    "SCH4": "Obligations of the tenant (Schedule 4, see Section 12(4))",
}


def pick_title(section, titles, margin_titles=None):
    """Choose the best available title for a section."""
    number = section["num"]

    if number in SCHEDULE_TITLES:
        return SCHEDULE_TITLES[number]

    if margin_titles and number in margin_titles:
        return margin_titles[number]

    if number in titles:
        return titles[number]

    if section.get("inline_title"):
        return section["inline_title"]

    words = section["text"].split()

    return " ".join(words[:8]) + ("..." if len(words) > 8 else "")


# ---------------------------------------------------------
# Chunking
# ---------------------------------------------------------

def chunk_text(text, max_chars=MAX_CHUNK_CHARS):
    """Split long section text into smaller chunks."""
    if len(text) <= max_chars:
        return [text]

    parts = re.split(r"(?=\s\(\d+\)\s)", text)

    chunks = []
    buffer = ""

    for part in parts:
        if buffer and len(buffer) + len(part) > max_chars:
            chunks.append(buffer.strip())
            buffer = part
        else:
            buffer += part

    if buffer.strip():
        chunks.append(buffer.strip())

    final_chunks = []

    for chunk in chunks:
        if len(chunk) <= max_chars * 1.5:
            final_chunks.append(chunk)
            continue

        sentences = re.split(r"(?<=[.;:])\s+", chunk)
        buffer = ""

        for sentence in sentences:
            if buffer and len(buffer) + len(sentence) > max_chars:
                final_chunks.append(buffer.strip())
                buffer = sentence
            else:
                buffer += " " + sentence

        if buffer.strip():
            final_chunks.append(buffer.strip())

    return final_chunks


# ---------------------------------------------------------
# Definitions parsing
# ---------------------------------------------------------

DEF_CLAUSE = re.compile(
    r"(?:(?<=\s)|^)\((\d{1,3})\)\s+[“\"]"
)


def split_definitions(text):
    """
    Split a definitions section into:
        [(clause_number, term, clause_text), ...]
    """

    hits = []
    last_number = 0

    for match in DEF_CLAUSE.finditer(text):
        number = int(match.group(1))

        if last_number < number <= last_number + 2:
            hits.append((number, match.start()))
            last_number = number

    if len(hits) < 5:
        return []

    items = []

    for index, (number, start) in enumerate(hits):
        if index + 1 < len(hits):
            end = hits[index + 1][1]
        else:
            end = len(text)

        body = text[start:end].strip()

        term_match = re.search(
            r"[“\"]([^”\"]{1,80})[”\"]",
            body,
        )

        term = term_match.group(1).strip() if term_match else ""

        items.append((number, term, body))

    return items


# ---------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------

def report_gaps(sections):
    numbers = sorted(
        {
            parse_num(section["num"])[0]
            for section in sections
            if not section["num"].startswith("SCH")
        }
    )

    if not numbers:
        return

    missing = [
        number
        for number in range(1, numbers[-1] + 1)
        if number not in numbers
    ]

    print(f"  Highest section number: {numbers[-1]}")

    if missing:
        shown = ", ".join(map(str, missing[:40]))

        if len(missing) > 40:
            shown += " ..."

        print(f"  Missing numbers ({len(missing)}): {shown}")

    else:
        print("  No gaps in numbering.")


def report_longest(sections):
    longest = sorted(
        sections,
        key=lambda section: len(section["text"]),
        reverse=True,
    )[:3]

    info = ", ".join(
        f"{section['num']} ({len(section['text'])} chars)"
        for section in longest
    )

    print(f"  Longest sections: {info}")


# ---------------------------------------------------------
# Create output records
# ---------------------------------------------------------

def add_chunks(records, cfg, record_id, section_label, title, text):
    chunks = chunk_text(text)

    for index, chunk in enumerate(chunks, start=1):
        suffix = "" if len(chunks) == 1 else f"-p{index}"

        records.append(
            {
                "id": f"{cfg['prefix']}-{record_id}{suffix}",
                "act": cfg["act"],
                "section": section_label,
                "title": title,
                "domain": cfg["domain"],
                "text": chunk,
            }
        )


def process_act(cfg):
    if cfg.get("skip"):
        print(
            "  [skipped on purpose: source PDF needs a cleaner copy first]"
        )
        return []

    pdf_path = RAW_DIR / cfg["file"]

    if not pdf_path.exists():
        print(
            f"  [skip] {cfg['file']} not found in data/raw/"
        )
        return []

    print(f"  Reading {cfg['file']} ...")

    margin_titles = {}

    if cfg.get("margin"):
        lines, margin_titles, warnings = read_pdf_positional(
            pdf_path,
            cfg["act"],
        )

        print(
            f"  Margin titles matched: {len(margin_titles)}, "
            f"unmatched notes: {len(warnings)}"
        )

        for warning in warnings[:15]:
            print("   ", warning)

    else:
        lines = read_pdf_lines(pdf_path, cfg["act"])

    total_chars = sum(len(line) for line in lines)

    if total_chars < 500:
        print(
            f"  [WARNING] Only {total_chars} characters found. "
            "Probably a scanned PDF (needs OCR)."
        )
        return []

    raw_sections, titles = split_into_sections(
        lines,
        cfg.get("schedules", False),
        cfg.get("ocr", False),
    )

    sections = dedupe_keep_longest(raw_sections)

    print(f"  Found {len(sections)} sections")

    report_gaps(sections)
    report_longest(sections)

    if not sections:
        print(
            "  [WARNING] No section headings matched. "
            "Paste me a few lines of this PDF."
        )

    if cfg.get("margin"):
        sections_without_titles = [
            section["num"]
            for section in sections
            if (
                section["num"] not in margin_titles
                and not section["num"].startswith("SCH")
            )
        ]

        print(
            "  Sections without a margin title "
            f"({len(sections_without_titles)}): "
            f"{', '.join(sections_without_titles[:40])}"
        )

    label = cfg.get("label", "Section")
    records = []

    for section in sections:
        if len(section["text"]) < 20:
            continue

        title = pick_title(
            section,
            titles,
            margin_titles,
        )

        is_schedule = section["num"].startswith("SCH")

        if is_schedule:
            section_label = f"Schedule {section['num'][3:]}"
        else:
            section_label = f"{label} {section['num']}"

        # Split Section 2 of the Consumer Protection Act into definitions.
        if cfg.get("definitions") == section["num"]:
            definitions = split_definitions(section["text"])

            if definitions:
                print(
                    f"  Definitions: split section {section['num']} "
                    f"into {len(definitions)} clauses "
                    f"(last clause number {definitions[-1][0]})"
                )

                for number, term, body in definitions:
                    if term:
                        definition_title = f"Definition: “{term}”"
                    else:
                        definition_title = f"Definition clause {number}"

                    add_chunks(
                        records,
                        cfg,
                        f"{section['num']}-cl{number}",
                        f"{label} {section['num']}({number})",
                        definition_title,
                        body,
                    )

                continue

            print(
                f"  [WARNING] Could not split section {section['num']} "
                "into definition clauses; using normal chunking."
            )

        add_chunks(
            records,
            cfg,
            section["num"],
            section_label,
            title,
            section["text"],
        )

    print(f"  Kept {len(records)} chunks")

    return records


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_records = []

    for cfg in ACTS:
        print(f"\n{cfg['act']}")
        all_records.extend(process_act(cfg))

    with open(OUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            all_records,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nWrote {len(all_records)} chunks to {OUT_FILE}")

    if all_records:
        print("\nSanity check, first 5 chunks:")

        for record in all_records[:5]:
            print(
                f"  {record['id']:12} | "
                f"{record['section']:12} | "
                f"{record['title'][:50]} | "
                f"{len(record['text'])} chars"
            )


if __name__ == "__main__":
    main()