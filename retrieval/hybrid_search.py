"""
Step 2 of the pipeline: given a question, find the most relevant law section(s).
Combines BM25 (keyword search) + vector search (semantic/multilingual) using
Reciprocal Rank Fusion (RRF).
"""

from pathlib import Path
import pickle

import chromadb
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE / "data" / "chroma_store"
BM25_FILE = BASE / "data" / "bm25_index.pkl"

EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model = None  # loaded once, reused


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
        _model.max_seq_length = 256  # must match build_index.py
    return _model


def load_bm25():
    with open(BM25_FILE, "rb") as f:
        return pickle.load(f)


def bm25_search(query, bm25_data, top_k=5):
    bm25 = bm25_data["bm25"]
    sections = bm25_data["sections"]
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(zip(sections, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [s["id"] for s, score in ranked]


def vector_search(query, top_k=5):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection("lexirag_sections")

    model = get_model()
    query_vec = model.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_vec], n_results=top_k)
    return results["ids"][0]


def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(query, top_k=3):
    bm25_data = load_bm25()

    # pull a wider candidate pool now that the corpus is bigger
    bm25_ranked = bm25_search(query, bm25_data, top_k=10)
    vector_ranked = vector_search(query, top_k=10)

    fused = reciprocal_rank_fusion([bm25_ranked, vector_ranked])
    top_ids = [doc_id for doc_id, score in fused[:top_k]]

    id_to_section = {s["id"]: s for s in bm25_data["sections"]}
    return [id_to_section[i] for i in top_ids]


if __name__ == "__main__":
    test_queries = [
        "my landlord isn't returning my deposit",
        "मेरी सैलरी नहीं मिली",
        "landlord returned deposit on time, can I still complain?",
    ]
    for q in test_queries:
        print(f"\nQUERY: {q}")
        results = hybrid_search(q)
        for r in results:
            print(f"  -> [{r['act']} - {r['section']}] {r['title']}")
