"""
LexiRAG: Hybrid retrieval for Indian legal documents.

Combines:
1. BM25 keyword search
2. Multilingual vector search
3. Reciprocal Rank Fusion
4. Synonym-aware lexical reranking
5. Optional cross-encoder re-ranking (rerank=True or LEXIRAG_RERANK=1)
6. Section-level deduplication
"""

from pathlib import Path
import os
import pickle
import re

import chromadb
from sentence_transformers import SentenceTransformer


BASE = Path(__file__).resolve().parent.parent

CHROMA_DIR = BASE / "data" / "chroma_store"
BM25_FILE = BASE / "data" / "bm25_index.pkl"

EMBED_MODEL_NAME = "BAAI/bge-m3"
COLLECTION_NAME = "lexirag_sections"

_model = None


STOP_WORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be",
    "been", "being", "i", "me", "my", "we", "our", "you", "your",
    "he", "she", "it", "they", "them", "this", "that", "these",
    "those", "do", "does", "did", "can", "could", "would",
    "should", "will", "shall", "may", "might", "to", "of", "in",
    "on", "at", "for", "from", "with", "by", "and", "or", "but",
    "if", "then", "what", "when", "where", "who", "how", "why",
    "have", "has", "had", "after", "before",
}


# Normalize common variations and synonyms into shared concepts.
SYNONYMS = {
    "refund": {
        "refund", "refunds", "refunded", "refunding",
        "return", "returns", "returned", "returning",
        "repay", "repays", "repaid", "repaying",
        "reimburse", "reimbursed", "reimbursement",
    },
    "deposit": {
        "deposit", "deposits",
    },
    "landlord": {
        "landlord", "landlords", "lessor", "lessors",
    },
    "tenant": {
        "tenant", "tenants", "lessee", "lessees",
    },
    "termination": {
        "end", "ends", "ended", "ending",
        "closure", "closed", "termination", "terminated",
        "outgoing", "expiry", "expired",
    },
    "salary": {
        "salary", "salaries", "wage", "wages", "pay", "payment",
        "remuneration",
    },
    "deduction": {
        "deduction", "deductions", "deduct", "deducted",
        "withheld", "withholding", "cut", "cuts",
    },
    "complaint": {
        "complaint", "complaints", "complain", "complaining",
        "grievance", "grievances",
    },
}


def get_model():
    """Load the embedding model once per Python process."""
    global _model

    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
        _model.max_seq_length = 512

    return _model


_bm25_data = None
_collection = None

# The "deposit + refund" bonus below was added by looking at a few test queries. Measured
# without it, every retrieval number was identical, so it is OFF by default (less overfitting).
# Turn it back on to reproduce the old behaviour:   $env:LEXIRAG_DEPOSIT_BONUS = "1"
USE_DEPOSIT_BONUS = os.getenv("LEXIRAG_DEPOSIT_BONUS", "0") == "1"


def load_bm25():
    """Load the BM25 index once per Python process."""
    global _bm25_data

    if _bm25_data is None:
        with open(BM25_FILE, "rb") as f:
            _bm25_data = pickle.load(f)

    return _bm25_data


def get_collection():
    """Open the ChromaDB collection once per Python process."""
    global _collection

    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)

    return _collection


def tokenize(text):
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def normalize_token(token):
    """Map synonyms to a shared searchable concept."""
    for canonical, variations in SYNONYMS.items():
        if token in variations:
            return canonical

    return token


def meaningful_terms(text):
    """Return normalized, meaningful terms."""
    return {
        normalize_token(token)
        for token in tokenize(text)
        if token not in STOP_WORDS and len(token) > 1
    }


def bm25_search(query, bm25_data, top_k=20, domain=None):
    """Retrieve candidate document IDs using BM25 (optionally one domain only)."""
    bm25 = bm25_data["bm25"]
    sections = bm25_data["sections"]

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    pairs = zip(sections, scores)
    if domain:
        pairs = [(sec, sc) for sec, sc in pairs if sec.get("domain") == domain]

    ranked = sorted(
        pairs,
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]

    return [
        section["id"]
        for section, score in ranked
        if score > 0
    ]


def vector_search(query, top_k=20, domain=None):
    """Retrieve candidate document IDs using semantic similarity (optionally one domain only)."""
    collection = get_collection()

    query_vector = get_model().encode([query]).tolist()[0]

    kwargs = {}
    if domain:
        kwargs["where"] = {"domain": domain}

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        **kwargs,
    )

    return results["ids"][0]


def reciprocal_rank_fusion(ranked_lists, k=60):
    """Combine rankings using Reciprocal Rank Fusion."""
    scores = {}

    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = (
                scores.get(doc_id, 0)
                + 1.0 / (k + rank + 1)
            )

    return sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )


def lexical_relevance(query, section):
    """
    Reward meaningful query concepts found in the title and text.

    Synonym normalization helps match natural-language questions
    to differently worded legal provisions.
    """
    query_terms = meaningful_terms(query)

    if not query_terms:
        return 0.0

    title = section.get("title", "")
    text = section.get("text", "")

    title_terms = meaningful_terms(title)
    text_terms = meaningful_terms(text)

    title_matches = query_terms & title_terms
    text_matches = query_terms & text_terms

    title_coverage = len(title_matches) / len(query_terms)
    text_coverage = len(text_matches) / len(query_terms)

    # Stronger lexical influence than the previous implementation.
    score = (
        0.06 * title_coverage
        + 0.12 * text_coverage
    )

    # Reward exact phrases from the question when they appear in
    # the provision. This is especially useful for legal terminology.
    query_tokens = tokenize(query)
    text_lower = text.lower()
    title_lower = title.lower()

    query_phrases = {
        " ".join(query_tokens[i:i + 2])
        for i in range(len(query_tokens) - 1)
        if query_tokens[i] not in STOP_WORDS
        and query_tokens[i + 1] not in STOP_WORDS
    }

    for phrase in query_phrases:
        if phrase in text_lower:
            score += 0.025
        elif phrase in title_lower:
            score += 0.04

    # Direct remedy + subject match:
    # e.g. refund/return + security deposit.
    if (
        USE_DEPOSIT_BONUS
        and "deposit" in query_terms
        and "refund" in query_terms
        and "deposit" in text_terms
        and "refund" in text_terms
    ):
        score += 0.10

    return score


def hybrid_search(query, top_k=3, rerank=None, pool=None, domain=None):
    """
    Retrieve legal sections, rerank them, and return at most one chunk
    per Act section.

    rerank=True  -> run the cross-encoder over the top `pool` chunks.
    rerank=None  -> follow the LEXIRAG_RERANK environment variable (1 = on).
    domain       -> search only this domain (consumer, labour, tenancy, legal_aid).
    """
    if rerank is None:
        rerank = os.getenv("LEXIRAG_RERANK", "0") == "1"
    if pool is None:
        pool = int(os.getenv("RERANK_POOL", "30"))

    bm25_data = load_bm25()
    sections = bm25_data["sections"]

    bm25_ranked = bm25_search(
        query,
        bm25_data,
        top_k=20,
        domain=domain,
    )

    vector_ranked = vector_search(
        query,
        top_k=20,
        domain=domain,
    )

    fused = reciprocal_rank_fusion(
        [bm25_ranked, vector_ranked]
    )

    id_to_section = {
        section["id"]: section
        for section in sections
    }

    reranked = []

    for doc_id, rrf_score in fused:
        section = id_to_section.get(doc_id)

        if section is None:
            continue

        final_score = (
            rrf_score
            + lexical_relevance(query, section)
        )

        reranked.append((section, final_score))

    reranked.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    ordered = [section for section, _ in reranked]

    if rerank and ordered:
        try:
            from retrieval.rerank import rerank as cross_rerank
        except ImportError:
            from rerank import rerank as cross_rerank

        # Copy each section and attach its cross-encoder score as "_score",
        # so the defer gate can judge confidence.
        ordered = [
            {**section, "_score": score}
            for section, score in cross_rerank(query, ordered[:pool])
        ]

    # Keep the highest-ranked chunk for each legal section.
    results = []
    seen_sections = set()

    for section in ordered:
        section_key = (
            section.get("act", ""),
            section.get("section", ""),
        )

        if section_key in seen_sections:
            continue

        seen_sections.add(section_key)
        results.append(section)

        if len(results) >= top_k:
            break

    return results


def routed_search(query, top_k=3, rerank=None, domains=None):
    """
    Search with domain routing.

    domains = None or []  -> normal search over everything
    one domain            -> search only that domain
    several domains       -> search each domain separately and interleave the results
                             (best of domain 1, best of domain 2, second of domain 1, ...)
    """
    if not domains:
        return hybrid_search(query, top_k=top_k, rerank=rerank)
    if len(domains) == 1:
        return hybrid_search(query, top_k=top_k, rerank=rerank, domain=domains[0])

    per_domain = -(-top_k // len(domains))  # ceiling division
    lists = [
        hybrid_search(query, top_k=per_domain, rerank=rerank, domain=d)
        for d in domains
    ]
    merged, seen = [], set()
    for rank in range(per_domain):
        for results in lists:
            if rank < len(results):
                section = results[rank]
                key = (section.get("act", ""), section.get("section", ""))
                if key not in seen:
                    seen.add(key)
                    merged.append(section)
    return merged[:top_k]


if __name__ == "__main__":
    test_queries = [
        "My landlord isn't returning my security deposit",
        "मेरी सैलरी नहीं मिली",
        "Landlord returned deposit on time, can I still complain?",
        "Employer cut my salary without telling me",
        "I bought a defective phone and the shop won't refund me",
    ]

    for query in test_queries:
        print(f"\nQUERY: {query}")

        results = hybrid_search(query, top_k=3)

        for section in results:
            print(
                f"  -> [{section['act']} - "
                f"{section['section']}] "
                f"{section['title']} "
                f"({section['id']})"
            )