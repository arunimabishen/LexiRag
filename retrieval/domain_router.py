"""
Domain router for LexiRAG: which area(s) of law does a question involve?

One small Groq call returns a list drawn from:
  tenancy   (Chhattisgarh Rent Control Act)
  consumer  (Consumer Protection Act)
  labour    (Payment of Wages Act)
  legal_aid (Legal Services Authorities Act)

Used to search only the relevant Act(s). A question with two areas is searched in
both and the results are interleaved. Returns None if the model cannot be reached
(callers then search everything). Cached in data/domain_cache.json.
"""
import json
import os
import re
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CACHE_FILE = BASE / "data" / "domain_cache.json"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

DOMAINS = ["tenancy", "consumer", "labour", "legal_aid"]

SYSTEM_PROMPT = """
You route legal questions to the right area of Indian law. The areas are:

- tenancy: landlord and tenant matters: rent, security deposit, eviction, tenancy
  agreements, rent controller, tenant and landlord duties.
- consumer: buying goods or services: defective goods, poor service, refunds,
  misleading advertisements, product liability, consumer complaints and commissions.
- labour: wages and salary: late payment, deductions, fines, claims for unpaid
  wages, wage penalties, appeals about wage claims.
- legal_aid: free legal help: who is eligible, legal services authorities,
  Lok Adalats. Use this ONLY when the person asks about free legal help or those
  bodies themselves.

Choose every area whose law is needed to answer the question, most important
first. Most questions involve exactly one area. If the question does not clearly
fit any area, return an empty list.

Reply with ONLY a JSON object, no other text:
{"domains": ["<area>", ...]}
""".strip()

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


def parse_domains(text):
    """Pull the domain list out of the model's reply; None if unusable."""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    raw = data.get("domains")
    if not isinstance(raw, list):
        return None
    out = []
    for d in raw:
        d = str(d).strip().lower()
        if d in DOMAINS and d not in out:
            out.append(d)
    return out


def route_domains(query):
    """Return a list of domains (possibly empty), or None on failure."""
    global _reasoning_ok
    cache = _load_cache()
    if query in cache:
        return cache[query]
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
            domains = parse_domains(response.choices[0].message.content)
            if domains is None:
                return None
            cache[query] = domains
            _save_cache()
            time.sleep(2.2)  # stay under the free-tier request limit
            return domains
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if _reasoning_ok and "reasoning" in msg:
                _reasoning_ok = False
                continue
            if "429" in msg or "rate" in msg:
                time.sleep(3 * (2 ** attempt))
                continue
            print(f"  router error: {type(e).__name__}: {e}")
            return None
    return None