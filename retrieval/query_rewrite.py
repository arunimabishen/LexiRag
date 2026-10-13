"""
Query rewriting for Hindi and Hinglish questions.

The legal text is in English, so a Hindi or Hinglish question is first rewritten
into one clear English question, and the SEARCH uses that. The user's original
question is still what the answer is written for.

Returns the English question, or None if the model could not be reached (callers
then fall back to searching with the original text). Results are cached in
data/rewrite_cache.json so repeated runs make no API calls.
"""
import json
import os
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CACHE_FILE = BASE / "data" / "rewrite_cache.json"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """
You rewrite questions for a search engine over Indian statutes about tenancy,
consumer protection, wages, and free legal aid.

The user's question may be in Hindi (Devanagari), Hinglish (Hindi written in
Roman letters), or English. Rewrite it as ONE clear English question.

Rules:
- Keep every fact and every negation exactly as the user stated them.
- Use plain, standard English words for the situation.
- Everyday words and the words these Acts use often differ, so where they do,
  write both in the question. Examples: "salary (wages)", "boss or company
  (employer)", "money cut or kept back (deducted)", "not paid on time (unpaid,
  delayed wages)", "shopkeeper (seller)", "money back (refund)", "house owner
  (landlord)", "free lawyer (free legal aid)".
- Do not answer the question. Do not add facts, Act names, or section numbers.
- Output only the English question, nothing else.
""".strip()

PROMPT_VERSION = 2  # bump when SYSTEM_PROMPT changes, so old cached rewrites are not reused

_cache = None
_reasoning_ok = True


def _load_cache():
    global _cache
    if _cache is None:
        if CACHE_FILE.exists():
            try:
                _cache = json.load(open(CACHE_FILE, encoding="utf-8"))
            except json.JSONDecodeError:
                _cache = {}
        else:
            _cache = {}
    return _cache


def _save_cache():
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(_cache, open(CACHE_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def rewrite_query(query):
    global _reasoning_ok
    cache = _load_cache()
    key = f"v{PROMPT_VERSION}|{query}"
    if key in cache:
        return cache[key]
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq
    except ImportError:
        return None

    client = Groq(api_key=GROQ_API_KEY)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}]
    for attempt in range(5):
        try:
            kwargs = dict(model=GROQ_MODEL, messages=messages, temperature=0,
                          max_completion_tokens=800)
            if _reasoning_ok:
                kwargs["reasoning_effort"] = "low"
            response = client.chat.completions.create(**kwargs)
            text = (response.choices[0].message.content or "").strip().strip('"').strip()
            if not text:
                return None
            cache[key] = text
            _save_cache()
            time.sleep(2.2)  # stay under the free-tier request limit
            return text
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if _reasoning_ok and "reasoning" in msg:
                _reasoning_ok = False
                continue
            if "429" in msg or "rate" in msg:
                time.sleep(3 * (2 ** attempt))
                continue
            print(f"  rewrite error: {type(e).__name__}: {e}")
            return None
    return None