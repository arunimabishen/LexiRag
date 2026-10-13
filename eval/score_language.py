r"""
Check language detection against the `lang` labels in both question files.
No models or API needed:   python eval\score_language.py
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from generation.language import detect_language  # noqa: E402

total = ok = 0
for name in ("eval_set.json", "gate_holdout.json"):
    path = BASE / "eval" / name
    if not path.exists():
        continue
    for q in json.load(open(path, encoding="utf-8")):
        total += 1
        got = detect_language(q["question"])
        if got == q["lang"]:
            ok += 1
        else:
            print(f"WRONG {q['id']}: wanted {q['lang']}, got {got} | {q['question']}")
print(f"\nLanguage detection: {ok}/{total} correct")