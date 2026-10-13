"""
Scope check for LexiRAG: one small LLM call that decides, from the question alone,
(1) is it inside the four Acts we cover, and (2) is it risky (criminal, violence,
urgent eviction, self-harm). Used by the defer-to-human gate.

Returns {"in_scope": bool, "risk": "none|criminal|violence|urgent_eviction|self_harm|other",
         "reason": str}, or None if the model could not be reached (the caller then
falls back to the keyword rules).
"""
import json
import os
import re
import time

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

RISKS = {"none", "criminal", "violence", "urgent_eviction", "self_harm", "other"}

SYSTEM_PROMPT = """
You are the scope checker for LexiRAG, an Indian legal information assistant.
LexiRAG can ONLY answer questions that are governed by these four Acts:

1. Chhattisgarh Rent Control Act, 2011: tenancy between landlord and tenant
   (rent, security deposit, eviction grounds and procedure, tenant and landlord
   duties, Rent Controller, appeals).
2. Consumer Protection Act, 2019: complaints about defective goods, deficiency
   in services, unfair trade practices, misleading advertisements, product
   liability, consumer commissions and their orders.
3. Payment of Wages Act, 1936: timely payment of wages, deductions and fines
   from wages, claims for delayed or unlawfully deducted wages, penalties.
4. Legal Services Authorities Act, 1987: who can get free legal aid, legal
   services authorities, Lok Adalats.

Decide two things about the user's question (it may be English, Hindi, or
Hinglish):

in_scope: true only if the main legal issue is governed by one of the four Acts
above. A question about any other area of law is false, even if it sounds legal.
When unsure, answer false.

risk: one of
- "none": ordinary civil question.
- "criminal": arrest, police, FIR, prosecution, bail, or any criminal matter.
- "violence": domestic violence, abuse, threats to physical safety, dowry harassment.
- "urgent_eviction": the person is being evicted or locked out right now.
- "self_harm": the person mentions suicide or harming themselves.
- "other": any other emergency or sensitive situation needing a human immediately.

Reply with ONLY a JSON object, no other text:
{"in_scope": true or false, "risk": "<one of the values above>", "reason": "<max 15 words>"}
""".strip()

_reasoning_ok = True


def parse_scope(text):
    """Pull the JSON object out of the model's reply and validate it."""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    in_scope = data.get("in_scope")
    if isinstance(in_scope, str):
        in_scope = in_scope.strip().lower() == "true"
    if not isinstance(in_scope, bool):
        return None
    risk = str(data.get("risk", "none")).strip().lower()
    if risk not in RISKS:
        risk = "other"
    return {"in_scope": in_scope, "risk": risk, "reason": str(data.get("reason", ""))[:200]}


def check_scope(query):
    global _reasoning_ok
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq
    except ImportError:
        return None

    client = Groq(api_key=GROQ_API_KEY)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    for attempt in range(5):
        try:
            kwargs = dict(model=GROQ_MODEL, messages=messages, temperature=0,
                          max_completion_tokens=1500)
            if _reasoning_ok:
                kwargs["reasoning_effort"] = "low"
            response = client.chat.completions.create(**kwargs)
            return parse_scope(response.choices[0].message.content)
        except Exception as e:  # noqa: BLE001
            text = str(e).lower()
            if _reasoning_ok and "reasoning" in text:
                _reasoning_ok = False  # this model/SDK does not accept the option
                continue
            if "429" in text or "rate" in text:
                time.sleep(3 * (2 ** attempt))
                continue
            print(f"  scope check error: {type(e).__name__}: {e}")
            return None
    return None