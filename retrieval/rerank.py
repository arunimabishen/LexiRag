"""
Cross-encoder re-ranker for LexiRAG.

Reads the question and each candidate section TOGETHER and scores how well the
section answers the question. Slower than vector search, so it is only run on
the top candidates from hybrid search.

Default model: BAAI/bge-reranker-v2-m3 (multilingual, handles Hindi/Hinglish).
Change it without editing code:   setx RERANK_MODEL "your/model-name"
"""
import os

RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

_cross_encoder = None


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(RERANK_MODEL, max_length=512)
    return _cross_encoder


def doc_text(section):
    return f"{section['act']} {section['section']}. {section['title']}. {section['text']}"


def rerank(query, sections):
    """Return [(section, score), ...] sorted best first."""
    if not sections:
        return []
    pairs = [(query, doc_text(s)) for s in sections]
    scores = get_cross_encoder().predict(pairs, batch_size=8, show_progress_bar=False)
    return sorted(zip(sections, [float(x) for x in scores]), key=lambda item: item[1], reverse=True)