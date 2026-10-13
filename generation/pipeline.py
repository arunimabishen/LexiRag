r"""
LexiRAG pipeline: question in, structured answer out.

    detect language -> gate (rules, then LLM scope check) -> rewrite Hindi/Hinglish
    question to English for search -> route to the right Act(s) -> retrieve + re-rank -> grounded answer in the
    user's language -> citations

Usage (from the project root):
    python generation\pipeline.py "My landlord isn't returning my deposit"
    python generation\pipeline.py            (runs a few demo questions)

Environment: GROQ_API_KEY (answers + scope check), and for the small re-ranker
    $env:RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    $env:RERANK_POOL  = "20"
"""
import json
import os
import re
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
try:  # print Hindi correctly even when output is piped (Windows would use cp1252)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from generation.gate import rule_hits, should_defer  # noqa: E402
from generation.generate import (  # noqa: E402
    GROQ_API_KEY, SYSTEM_PROMPT, TOP_K, build_prompt, call_llm, get_field, mock_llm,
)
from generation.language import detect_language  # noqa: E402
from generation.scope_check import check_scope  # noqa: E402
from retrieval.domain_router import route_domains  # noqa: E402
from retrieval.hybrid_search import routed_search  # noqa: E402
from retrieval.query_rewrite import rewrite_query  # noqa: E402

# If the scope check cannot be reached (and a Groq key IS configured), refuse to answer
# rather than risk answering an out-of-scope or risky question.
FAIL_CLOSED = True

# Domain routing (search only the relevant Act) was measured to add only noise-level gains,
# so it is OFF by default. Turn it on with:   $env:LEXIRAG_ROUTING = "1"
USE_ROUTING = os.getenv("LEXIRAG_ROUTING", "0") == "1"

# Verified on nalsa.gov.in and telemanas.mohfw.gov.in. Add the Durg District Legal
# Services Authority here ONLY after copying it from an official NALSA / CGSLSA page.
LOCAL_DLSA = None  # e.g. {"name": "DLSA Durg", "contact": "..."}

NALSA = {"name": "NALSA free legal aid helpline", "contact": "15100 (toll free)"}
DLSA_GENERIC = {"name": "District Legal Services Authority (DLSA)",
                "contact": "Office in your district court complex; apply on plain paper"}
TELEMANAS = {"name": "Tele-MANAS mental health helpline", "contact": "14416 or 1-800-891-4416 (24x7)"}
EMERGENCY = {"name": "Emergency services (police)", "contact": "112"}

DEFER_TEXT = {
    "en": ("I can't safely answer this one. It is outside what I cover, or it needs a person to look at "
           "your situation. Please contact free legal aid: call the NALSA helpline on 15100 or visit your "
           "District Legal Services Authority."),
    "hi": ("मैं इस सवाल का सुरक्षित उत्तर नहीं दे सकता। यह मेरे दायरे से बाहर है या इसमें किसी व्यक्ति को आपकी "
           "स्थिति देखनी होगी। कृपया मुफ़्त कानूनी सहायता से संपर्क करें: NALSA हेल्पलाइन 15100 पर कॉल करें या अपने "
           "ज़िला विधिक सेवा प्राधिकरण (DLSA) जाएँ।"),
    "hinglish": ("Main is sawaal ka surakshit jawab nahi de sakta. Ye mere daayre se bahar hai ya isme kisi insaan ko "
                 "aapki sthiti dekhni hogi. Kripya free legal aid se sampark karein: NALSA helpline 15100 par call "
                 "karein ya apne District Legal Services Authority (DLSA) jaayein."),
}
UNAVAILABLE_TEXT = {
    "en": "I could not complete the safety check just now, so I will not answer. Please try again in a minute, or contact free legal aid on 15100.",
    "hi": "मैं अभी सुरक्षा जाँच पूरी नहीं कर सका, इसलिए उत्तर नहीं दे रहा। कृपया एक मिनट बाद फिर कोशिश करें या 15100 पर मुफ़्त कानूनी सहायता से संपर्क करें।",
    "hinglish": "Main abhi safety check poora nahi kar saka, isliye jawab nahi de raha. Kripya thodi der baad phir try karein ya 15100 par free legal aid se sampark karein.",
}
GROUNDING_REMINDER = (
    "\n\nSTRICT: Mention only Acts, section numbers, authorities and offices that appear in the "
    "excerpts above. If something not in the excerpts could matter, say the full Act or a lawyer "
    "should be checked, without naming it."
)
LANGUAGE_INSTRUCTION = {
    "en": "",
    "hi": ("\n\nLANGUAGE: Write the entire answer in simple Hindi (Devanagari script). Keep Act names, "
           "section numbers and the [S1]-style markers exactly as given. Translate legal terms carefully "
           "(for example 'sanction' means permission to prosecute, not a punishment); if you are not "
           "sure of a Hindi term, keep the English term in brackets)."),
    "hinglish": ("\n\nLANGUAGE: Write the entire answer in Hinglish (Hindi written in Roman letters, the way "
                 "people text), keeping Act names, section numbers and the [S1]-style markers exactly as given."),
}


def resources_for(reasons):
    items = [NALSA, LOCAL_DLSA or DLSA_GENERIC]
    joined = " ".join(reasons)
    if "self_harm" in joined:
        items.insert(0, TELEMANAS)
    if any(k in joined for k in ("violence", "domestic", "criminal", "urgent_eviction", "eviction")):
        items.insert(0, EMERGENCY)
    return items


def deferral(query, language, reasons, start, extra=None):
    out = {
        "query": query, "language": language, "deferred": True,
        "defer_reason": reasons[0] if reasons else None, "defer_reasons": reasons,
        "answer": DEFER_TEXT[language], "citations": [], "resources": resources_for(reasons),
        "seconds": round(time.time() - start, 1),
    }
    if extra:
        out.update(extra)
    return out


def multilingual_system_prompt():
    new = SYSTEM_PROMPT.replace(
        "Use clear, concise, plain English.",
        "Use clear, concise, plain language, in the language the user message asks for (default English).",
    )
    if new == SYSTEM_PROMPT:  # the wording changed; fall back to an appended rule
        new += "\n\nWrite in the language the user message asks for (default English)."
    return new


def answer(query):
    start = time.time()
    language = detect_language(query)

    # 1) Gate, cheap rules first
    rules = rule_hits(query)
    if rules:
        return deferral(query, language, rules, start, {"gate": {"rules": rules, "scope": None}})

    # 2) Gate, LLM scope check
    scope = None
    if GROQ_API_KEY:
        scope = check_scope(query)
        if scope is None and FAIL_CLOSED:
            out = deferral(query, language, ["scope_check_unavailable"], start,
                           {"gate": {"rules": [], "scope": None}})
            out["answer"] = UNAVAILABLE_TEXT[language]
            return out
    decision = should_defer(query, scope=scope)
    if decision["defer"]:
        return deferral(query, language, decision["reasons"], start,
                        {"gate": {"rules": [], "scope": scope}})

    # 3) Retrieve (with cross-encoder re-ranking). The law is in English, so a Hindi or
    #    Hinglish question is searched through its English rewrite.
    search_query = query
    if language != "en":
        rewritten = rewrite_query(query)
        if rewritten:
            search_query = rewritten
    domains = route_domains(search_query) if USE_ROUTING else None
    k = TOP_K + (1 if domains and len(domains) > 1 else 0)
    sections = routed_search(search_query, top_k=k, rerank=True, domains=domains)
    if not sections:
        return deferral(query, language, ["no_sources_found"], start,
                        {"gate": {"rules": [], "scope": scope}})

    # 4) Grounded answer in the user's language
    user_prompt = build_prompt(query, sections) + GROUNDING_REMINDER + LANGUAGE_INSTRUCTION[language]
    text = call_llm(multilingual_system_prompt(), user_prompt) if GROQ_API_KEY else None
    if text is None:
        text = mock_llm(query, sections)

    # 5) Citations: every retrieved source, flagged if the answer actually used it
    # Models sometimes write 【S1】 (full-width brackets) in Hindi; accept both and normalise.
    marker = re.compile(r"[\[【]\s*S(\d+)\s*[\]】]")
    used = {int(n) for n in marker.findall(text)}
    text = marker.sub(lambda m: f"[S{m.group(1)}]", text)
    citations = []
    for i, s in enumerate(sections, start=1):
        citations.append({
            "marker": f"S{i}",
            "act": get_field(s, "act", "act_name"),
            "section": get_field(s, "section", "provision"),
            "title": get_field(s, "title"),
            "source_id": get_field(s, "id", "chunk_id"),
            "text": get_field(s, "text", "content", "page_content"),
            "cited": i in used,
        })

    return {
        "query": query, "language": language, "deferred": False, "defer_reason": None,
        "defer_reasons": [], "answer": text, "citations": citations, "resources": [],
        "gate": {"rules": [], "scope": scope}, "search_query": search_query, "domains": domains,
        "seconds": round(time.time() - start, 1),
    }


if __name__ == "__main__":
    questions = sys.argv[1:] or [
        "My landlord isn't returning my security deposit. What can I do?",
        "meri salary se company ne bina bataye paise kaat liye",
        "Police ne mere bhai ko arrest kar liya hai",
    ]
    for q in questions:
        print("=" * 78)
        print(json.dumps(answer(q), ensure_ascii=False, indent=2))