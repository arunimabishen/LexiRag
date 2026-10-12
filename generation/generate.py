"""
Step 3 of the pipeline: turn retrieved law text into a plain-language,
cited answer. This is the "RAG" payoff — the LLM is NOT allowed to answer
from its own memory, only from the real text we hand it.
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from retrieval.hybrid_search import hybrid_search

SYSTEM_PROMPT = """You are LexiRAG, a legal information assistant for Indian law.

STRICT RULES:
1. Answer ONLY using the retrieved sections provided below. Do not use outside knowledge.
2. Every claim you make must be traceable to one of the provided sections.
3. Always cite the exact Act name and Section number for each claim.
4. Structure your answer as:
   - A short plain-language explanation of the person's rights
   - A numbered action checklist (what to actually do next)
   - A "Source" line citing the Act + Section
5. If the retrieved sections do NOT clearly answer the question, say so explicitly
   and recommend contacting free legal aid (NALSA/DLSA) instead of guessing.
"""


def build_prompt(query, retrieved_sections):
    context = "\n\n".join(
        f"[{s['act']} - {s['section']}] {s['title']}\n{s['text']}"
        for s in retrieved_sections
    )
    return f"""Retrieved sections:
{context}

User question: {query}

Answer the user's question following the rules above."""


def call_llm(system_prompt, user_prompt):
    """Real call to Groq (free tier). Requires GROQ_API_KEY to be set."""
    from groq import Groq

    client = Groq()  # reads GROQ_API_KEY from env automatically
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def mock_llm(query, retrieved_sections):
    """
    No API key available right now -> produce a template-based answer so the
    pipeline is still fully demonstrable end-to-end.
    """
    top = retrieved_sections[0]
    lines = [
        "[MOCK ANSWER - no LLM API key set, this is a template stand-in]\n",
        f"Based on {top['act']}, {top['section']} ({top['title']}):",
        f"\"{top['text']}\"\n",
        "What to do next:",
        "1. Document the relevant dates and any communication with the other party.",
        f"2. Refer to {top['section']} of the {top['act']} when raising your claim.",
        "3. If unresolved, approach the relevant authority mentioned in the Act.",
        f"\nSource: {top['act']}, {top['section']}",
    ]
    return "\n".join(lines)


def answer_question(query):
    retrieved = hybrid_search(query, top_k=3)
    user_prompt = build_prompt(query, retrieved)

    if os.environ.get("GROQ_API_KEY"):
        answer = call_llm(SYSTEM_PROMPT, user_prompt)
    else:
        answer = mock_llm(query, retrieved)

    return {
        "query": query,
        "retrieved_sections": [f"{s['act']} {s['section']}" for s in retrieved],
        "answer": answer,
    }


if __name__ == "__main__":
    result = answer_question("my landlord isn't returning my deposit, what can I do?")
    print("QUERY:", result["query"])
    print("\nRETRIEVED:", result["retrieved_sections"])
    print("\nANSWER:\n", result["answer"])