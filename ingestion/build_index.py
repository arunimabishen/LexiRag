"""
Step 1 of the pipeline: turn law sections into a searchable index.

1. Load sections (real extracted Acts if present, otherwise the sample data)
2. Embed each section with a MULTILINGUAL model (Hindi/English/Hinglish)
3. Store vectors in ChromaDB (meaning-based search)
4. Build a BM25 index (exact keyword / section-number search)
"""

import json
import pickle
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

BASE = Path(__file__).resolve().parent.parent

# Use the real extracted Acts if they exist, otherwise fall back to sample data.
REAL_DATA = BASE / "data" / "processed" / "sections.json"
SAMPLE_DATA = BASE / "data" / "sample_sections.json"
DATA_FILE = REAL_DATA if REAL_DATA.exists() else SAMPLE_DATA

CHROMA_DIR = BASE / "data" / "chroma_store"
BM25_FILE = BASE / "data" / "bm25_index.pkl"

EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def load_sections():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vector_index(sections, model):
    print(f"Embedding {len(sections)} sections with {EMBED_MODEL_NAME} ...")
    texts = [f"{s['title']}. {s['text']}" for s in sections]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection("lexirag_sections")
    except Exception:
        pass
    collection = client.create_collection("lexirag_sections")

    collection.add(
        ids=[s["id"] for s in sections],
        embeddings=embeddings,
        documents=[s["text"] for s in sections],
        metadatas=[
            {
                "act": s["act"],
                "section": s["section"],
                "title": s["title"],
                "domain": s.get("domain", "general"),
            }
            for s in sections
        ],
    )
    print(f"Vector index built and saved to {CHROMA_DIR}")


def build_bm25_index(sections):
    print("Building BM25 keyword index ...")
    tokenized_corpus = [
        (s["title"] + " " + s["text"]).lower().split() for s in sections
    ]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_FILE, "wb") as f:
        pickle.dump({"bm25": bm25, "sections": sections}, f)
    print(f"BM25 index built and saved to {BM25_FILE}")


if __name__ == "__main__":
    print(f"Using data file: {DATA_FILE}")
    sections = load_sections()
    model = SentenceTransformer(EMBED_MODEL_NAME)
    # Default is 128 tokens, which would cut real legal chunks short.
    model.max_seq_length = 256
    build_vector_index(sections, model)
    build_bm25_index(sections)
    print("\nDone. Index is ready for querying.")
