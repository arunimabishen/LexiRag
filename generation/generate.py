import os
import sys
from pathlib import Path

# --------------------------------------------------
# 1. PROJECT PATH
# --------------------------------------------------

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from retrieval.hybrid_search import hybrid_search


# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Number of sources passed to the answer step. Default 3; test 5 with:  $env:LEXIRAG_TOP_K = "5"
TOP_K = int(os.getenv("LEXIRAG_TOP_K", "3"))


# --------------------------------------------------
# 3. LEGAL GROUNDING SYSTEM PROMPT
# --------------------------------------------------

SYSTEM_PROMPT = """
You are LexiRAG, a legal question-answering assistant for Indian law.

Answer using only the legal excerpts supplied in the current user message.
Those excerpts may be incomplete, may omit surrounding clauses, and may not
contain all the law relevant to the question.

NON-NEGOTIABLE GROUNDING RULES

1. SOURCE-BOUND LEGAL CLAIMS
- Make a claim about a legal right, duty, prohibition, remedy, penalty,
  procedure, deadline, authority, or jurisdiction only when the supplied text
  directly supports it.
- Do not use outside legal knowledge to fill gaps.
- Do not invent section numbers, quotations, requirements, exceptions, or
  legal outcomes.
- Do not infer who holds a right or duty unless the excerpt identifies that
  person or the wording unambiguously establishes it.
- Do not infer a filing route, authority, deadline, or procedure merely because
  an authority or application is mentioned.

2. PRECISE INTERPRETATION
- Preserve important conditions in the text, including time limits, qualifying
  circumstances, and who must do what.
- Do not turn a conditional rule into an unconditional right.
- Do not treat a possible remedy as a guaranteed result.
- Do not conclude that conduct is illegal from a fact that the excerpt does
  not establish. For example, a user saying they were not informed does not,
  by itself, prove that a wage deduction was unlawful.
- Do not expand a provision beyond its stated scope. For example, a provision
  about harm caused by a defective product does not establish that every
  defective product automatically qualifies for that action.

3. EXPLICITLY HANDLE GAPS
- If the excerpts do not establish an answer, say what is and is not clear.
- The absence of a rule in the excerpts does not prove that the full Act has
  no such rule or remedy.
- Do not guess to make the answer sound complete.
- When needed, say that the relevant surrounding provisions or complete Act
  must be checked, or suggest consulting a qualified Indian lawyer.

4. PRACTICAL NEXT STEPS
- You may suggest general, non-legal precautions such as preserving receipts,
  payslips, agreements, photographs, and written communications.
- Clearly label these as practical suggestions, not statutory requirements.
- Do not recommend a specific court, tribunal, government office, form, notice,
  filing deadline, or legal procedure unless the supplied excerpts support it.
- Only tell the user to take a legal step if an excerpt says the USER (or a
  person in the user's position) can take that step. If an excerpt describes a
  step for an authority, court or the other party (for example a show-cause
  opportunity given to an employer before a sanction), do not turn it into
  something the user should request or do.
- If the excerpts do not say what the user can do, say so plainly and suggest
  consulting free legal aid or a qualified lawyer, instead of inventing a step.

5. CITATIONS TO RETRIEVED SOURCES
- Each legal claim must end with a source marker such as [S1] or [S2].
- Use only the source markers and Act/provision details supplied in the context.
- Use a marker only if that source actually supports the claim.
- Do not cite a source merely because it concerns the same general topic.
- If no retrieved source supports a legal claim, do not state it as law.

RESPONSE STYLE
- Use clear, concise, plain English.
- Answer the user's actual question without unnecessary repetition.
- Use only the headings that are useful:
  1. What the retrieved law says
  2. What this may mean for you
  3. Practical next steps
  4. What remains uncertain
- Accuracy is more important than completeness. Do not present this response as
  a substitute for advice from a qualified lawyer.
""".strip()


# --------------------------------------------------
# 4. HELPER: READ FIELDS FROM RETRIEVED SECTIONS
# --------------------------------------------------

def get_field(section, *field_names, default=""):
    """Read a field from a result dictionary or its nested metadata."""
    if not isinstance(section, dict):
        return default

    metadata = section.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    for field in field_names:
        value = section.get(field)
        if value is None or value == "":
            value = metadata.get(field)
        if value is not None and value != "":
            return value

    return default


# --------------------------------------------------
# 5. BUILD THE USER PROMPT
# --------------------------------------------------

def build_prompt(query, retrieved_sections):
    context_parts = []

    for index, section in enumerate(retrieved_sections, start=1):
        act = get_field(section, "act", "act_name", default="Unknown Act")
        provision = get_field(
            section, "section", "provision", default="Unknown provision"
        )
        title = get_field(section, "title", default="")
        chunk_id = get_field(
            section, "id", "chunk_id", default="Unknown chunk"
        )
        legal_text = get_field(
            section, "text", "content", "page_content", default=""
        )

        context_parts.append(
            f"[S{index}]\n"
            f"Act: {act}\n"
            f"Provision: {provision}\n"
            f"Title: {title}\n"
            f"Source ID: {chunk_id}\n"
            f"Retrieved legal text:\n{legal_text}"
        )

    context = "\n\n".join(context_parts)

    return f"""
USER QUESTION:
{query}

RETRIEVED LEGAL EXCERPTS:
{context}

INSTRUCTIONS FOR THIS ANSWER:
1. First identify the exact propositions that the excerpts establish.
2. Preserve the conditions and limitations in each provision.
3. Attach a source marker such as [S1] or [S2] to every legal claim, using only the
   source that directly supports that claim.
4. Do not assume that a source gives a particular person a right unless the
   source text establishes who holds that right.
5. Distinguish a fact in the user's story from a legal conclusion. If the
   excerpts do not establish whether the conduct was unlawful, say so.
6. Do not add unverified legal processes or guaranteed outcomes.
7. Give only practical suggestions that are clearly labelled as such. Do not tell the
   user to ask for or do something that the excerpts describe as a step for an
   authority, a court or the other party.
8. State what cannot be determined from these excerpts and what text would need
   to be checked next.

Answer in plain English. Do not invent facts or legal rules.
""".strip()


# --------------------------------------------------
# 6. CALL GROQ
# --------------------------------------------------

def call_llm(system_prompt, user_prompt):
    if not GROQ_API_KEY:
        return None

    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_completion_tokens=2500,
        )

        answer = response.choices[0].message.content
        if not answer:
            return "The model returned an empty answer. Please try again."
        return answer.strip()

    except Exception as e:
        print(f"\nGroq API error: {type(e).__name__}: {e}")
        return (
            "I could not generate an answer because the language model request "
            "failed. Please check your Groq API configuration and try again."
        )


# --------------------------------------------------
# 7. FALLBACK WHEN GROQ API KEY IS MISSING
# --------------------------------------------------

def mock_llm(query, retrieved_sections):
    """Show retrieved passages without pretending to generate legal advice."""
    lines = [
        "Groq is not configured, so a generated legal answer is unavailable.",
        "",
        "Your question:",
        query,
        "",
        "Retrieved legal excerpts:",
    ]

    for index, section in enumerate(retrieved_sections, start=1):
        act = get_field(section, "act", "act_name", default="Unknown Act")
        provision = get_field(
            section, "section", "provision", default="Unknown provision"
        )
        title = get_field(section, "title", default="")
        chunk_id = get_field(
            section, "id", "chunk_id", default="Unknown chunk"
        )
        legal_text = get_field(
            section, "text", "content", "page_content", default=""
        )

        lines.extend([
            "",
            f"[S{index}] Act: {act}",
            f"Provision: {provision}",
            f"Title: {title}",
            f"Source ID: {chunk_id}",
            f"Text: {legal_text}",
        ])

    lines.extend([
        "",
        "Configure GROQ_API_KEY to generate an answer.",
        "These excerpts may be incomplete and are not necessarily a complete "
        "statement of the applicable law.",
    ])
    return "\n".join(lines)


# --------------------------------------------------
# 8. ANSWER A LEGAL QUESTION
# --------------------------------------------------

def answer_question(query):
    try:
        retrieved_sections = hybrid_search(query, top_k=TOP_K)
    except Exception as e:
        return {
            "query": query,
            "retrieved_sections": [],
            "answer": f"Retrieval failed: {type(e).__name__}: {e}",
        }

    if retrieved_sections is None:
        retrieved_sections = []

    if len(retrieved_sections) == 0:
        return {
            "query": query,
            "retrieved_sections": [],
            "answer": (
                "I could not retrieve relevant legal excerpts for your question. "
                "I cannot provide a source-grounded answer without relevant text. "
                "Try rephrasing your question or consult a qualified Indian lawyer."
            ),
        }

    user_prompt = build_prompt(query, retrieved_sections)

    if GROQ_API_KEY:
        answer = call_llm(SYSTEM_PROMPT, user_prompt)
        if answer is None:
            answer = (
                "I could not generate an answer because the language model "
                "request failed. Please check your Groq API configuration."
            )
    else:
        answer = mock_llm(query, retrieved_sections)

    return {
        "query": query,
        "retrieved_sections": retrieved_sections,
        "answer": answer,
    }


# --------------------------------------------------
# 9. PRINT RESULTS
# --------------------------------------------------

def print_result(result):
    print("\n" + "=" * 80)
    print(f"QUERY: {result['query']}")
    print("=" * 80)
    print("\nRETRIEVED SOURCES:\n")

    sections = result.get("retrieved_sections", [])
    if not sections:
        print("No sources retrieved.")

    for index, section in enumerate(sections, start=1):
        act = get_field(section, "act", "act_name", default="Unknown Act")
        provision = get_field(
            section, "section", "provision", default="Unknown provision"
        )
        title = get_field(section, "title", default="")
        chunk_id = get_field(
            section, "id", "chunk_id", default="Unknown chunk"
        )
        print(f"[S{index}] {act} | {provision} | {title} | {chunk_id}")

    print("\nANSWER:\n")
    print(result.get("answer", "No answer generated."))
    print()


# --------------------------------------------------
# 10. RUN TEST QUERIES
# --------------------------------------------------

if __name__ == "__main__":
    test_queries = [
        "My landlord isn't returning my security deposit. What can I do?",
        "My employer deducted my salary without telling me. What can I do?",
        "I bought a defective phone and the shop refuses to refund me. What can I do?",
    ]

    for query in test_queries:
        result = answer_question(query)
        print_result(result)