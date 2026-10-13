r"""
Language detection for LexiRAG: English ("en"), Hindi in Devanagari ("hi"),
Hinglish = Hindi written in Roman letters ("hinglish").

No model needed: Devanagari script check plus a list of common Romanized Hindi words.
Measure it with:  python eval\score_language.py
"""
import re

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
LATIN_WORD = re.compile(r"[a-zA-Z]+")

# Common Romanized-Hindi words that are NOT ordinary English words.
HINGLISH_WORDS = {
    "hai", "hain", "nahi", "nahin", "mera", "meri", "mere", "mujhe", "mujhse", "mujhko",
    "kya", "kaise", "kyun", "kyon", "karun", "karu", "kare", "karna", "kar", "kiya", "kiye",
    "raha", "rahi", "rahe", "wapas", "vapas", "paise", "paisa", "dukan", "dukaan", "malik",
    "hamare", "hamara", "hamari", "hum", "aap", "aur", "bina", "bataye", "bataya", "batao",
    "bataiye", "kaat", "liye", "diya", "diye", "mila", "mili", "mile", "milega", "milti",
    "kharab", "kharida", "gaya", "gayi", "hota", "hoga", "hogi", "chahiye", "sakta", "sakte",
    "sakti", "kab", "kahan", "kaun", "kitna", "kitne", "abhi", "aaj", "wale", "wali",
    "gaon", "gaadi", "bete", "beti", "pati", "patni", "ghar", "kaam", "hata", "nikal",
    "nikaal", "ka", "ki", "ke", "ko", "se", "mein", "tha", "thi", "the", "ho", "hua", "hui",
    "toh", "bhi", "lekin", "agar", "magar", "faisla", "muavza", "maarte", "maar",
    "nahi", "salary", "wapis", "dena", "dete", "deta", "deti", "lena", "leta", "leti",
}
# These also exist in English; only count them if other Hinglish words are present.
AMBIGUOUS = {"the", "ho", "se", "ko", "ka", "ke", "ki", "tha", "thi", "kar", "hum", "maar", "salary"}


def detect_language(text):
    """Return "hi", "hinglish" or "en"."""
    if not text or not text.strip():
        return "en"
    dev = len(DEVANAGARI.findall(text))
    latin = len(re.findall(r"[a-zA-Z]", text))
    if dev and dev >= 0.3 * (dev + latin):
        return "hi"
    words = [w.lower() for w in LATIN_WORD.findall(text)]
    strong = [w for w in words if w in HINGLISH_WORDS and w not in AMBIGUOUS]
    weak = [w for w in words if w in AMBIGUOUS and w in HINGLISH_WORDS]
    if len(strong) >= 2 or (len(strong) >= 1 and len(strong) + len(weak) >= 3):
        return "hinglish"
    if dev:  # a little Devanagari mixed into mostly Latin text
        return "hinglish"
    return "en"