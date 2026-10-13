"""
Defer-to-human gate for LexiRAG.

Decides whether to ANSWER or to DEFER to free legal aid (NALSA / DLSA).
Three layers:
  1. Rules: urgent or out-of-scope situations always defer (English, Hinglish, Hindi).
  2. Scope check (optional): an LLM verdict from generation/scope_check.py. Defers
     when the question is outside the four Acts or is risky.
  3. Confidence: if the best retrieved section scores below a threshold, defer.
     (Measured to be a weak signal; it is off by default.)

Usage:
    from generation.gate import should_defer
    from generation.scope_check import check_scope
    decision = should_defer(query, top_score, scope=check_scope(query))
    # {"defer": True/False, "reasons": ["domestic_violence", "out_of_scope_llm", ...]}
"""
import os
import re

# Raw cross-encoder score below this = "not confident". The right value comes from
# eval/score_gate.py (it sweeps thresholds on the eval set). Override without editing:
#   $env:LEXIRAG_CONF_THRESHOLD = "-3.5"
CONF_THRESHOLD = float(os.getenv("LEXIRAG_CONF_THRESHOLD", "-99"))  # -99 = confidence layer off

# (reason, [patterns]). Latin patterns are regex (case-insensitive); Hindi is matched as-is.
RULES = [
    ("self_harm", [
        r"\bsuicide\b", r"kill myself", r"end my life", r"\baatmahatya\b", r"jaan de(?:na|ne)",
        "आत्महत्या", "जान दे",
    ]),
    ("domestic_violence", [
        r"\bbeats? me\b", r"\bbeating me\b", r"domestic violence", r"\bdomestic abuse\b",
        r"\babus(?:e|es|ed|ing)\b", r"\bdowry\b", r"\bdahej\b", r"\bmaar(?:te|ta|ti|ne)?\s+(?:hain|hai|hain\.)",
        r"\bmaarpeet\b", r"\bmaar[- ]peet\b", r"\bharass(?:es|ed|ing|ment)?\b.*\b(husband|in-laws|wife)\b",
        "मारपीट", "घरेलू हिंसा", "दहेज", "मारते हैं", "मारता है",
    ]),
    ("criminal_matter", [
        r"\barrest(?:ed|s|ing)?\b", r"\bpolice\b", r"\bfir\b", r"\bjail\b", r"\bbail\b",
        r"\bchargesheet\b", r"\bmurder\b", r"\bcriminal case\b", r"\bgiraft(?:a|ar)",
        "गिरफ्तार", "पुलिस", "जेल", "जमानत",
    ]),
    ("eviction_in_progress", [
        r"\bbailiffs?\b", r"evict(?:ed|ing)? (?:me )?(?:today|now|tonight)",
        r"(?:today|right now|tonight).{0,40}\bevict", r"\babhi (?:ghar )?nikal", r"\baaj hi (?:ghar )?(?:khali|nikal)",
        "अभी निकाल", "आज ही घर खाली",
    ]),
    ("out_of_scope", [
        r"\bdivorce\b", r"\btalaq\b", r"\bcustody of\b", r"\balimony\b", r"\bmaintenance case\b",
        r"\bcheque\b", r"\bchec?k bounce", r"\bstocks?\b", r"\bshares?\b to buy", r"\bcrypto", r"\binvest(?:ment)?\b",
        r"\bincome tax\b", r"\bgst\b", r"\bwill and testament\b", r"\binherit(?:ance)?\b",
        "तलाक", "चेक बाउंस",
    ]),
]


def rule_hits(query):
    hits = []
    for reason, patterns in RULES:
        for p in patterns:
            if re.search(p, query, re.IGNORECASE):
                hits.append(reason)
                break
    return hits


def should_defer(query, top_score=None, threshold=None, scope=None):
    """Return {"defer": bool, "reasons": [...]}.

    scope = result of check_scope(query), or None if the LLM check was not run
    or could not be reached (then only the rules and the confidence layer apply).
    """
    reasons = rule_hits(query)
    if scope:
        if not scope.get("in_scope", True):
            reasons.append("out_of_scope_llm")
        risk = scope.get("risk", "none")
        if risk != "none":
            reasons.append(f"risk_{risk}")
    if threshold is None:
        threshold = CONF_THRESHOLD
    if not reasons and top_score is not None and top_score < threshold:
        reasons.append("low_confidence")
    return {"defer": bool(reasons), "reasons": reasons}