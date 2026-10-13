"""
Build or reuse LexiRAG's search indexes.

- ChromaDB: semantic/vector search
- BM25: keyword search
- Skip embedding generation when existing indexes match sections.json
"""

import json
import pickle
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


BASE = Path(__file__).resolve().parent.parent

REAL_DATA = BASE / "data" / "processed" / "sections.json"
SAMPLE_DATA = BASE / "data" / "sample_sections.json"
DATA_FILE = REAL_DATA if REAL_DATA.exists() else SAMPLE_DATA

CHROMA_DIR = BASE / "data" / "chroma_store"
BM25_FILE = BASE / "data" / "bm25_index.pkl"

EMBED_MODEL_NAME = "BAAI/bge-m3"
COLLECTION_NAME = "lexirag_sections"
MAX_SEQ_LENGTH = 512


def load_sections():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def index_text(section):
    """Text used for both semantic and keyword indexing."""
    return (
        f"{section['act']} "
        f"{section['section']}. "
        f"{section['title']}. "
        f"{section['text']}"
    )


def indexes_are_current(sections):
    """
    Reuse existing indexes if:
    1. The saved BM25 index contains the current sections.
    2. The Chroma collection contains exactly the expected chunk IDs.
    """
    if not BM25_FILE.exists():
        return False

    try:
        with open(BM25_FILE, "rb") as f:
            bm25_data = pickle.load(f)

        # Check whether the saved keyword index uses current source data.
        if bm25_data.get("sections") != sections:
            return False

        if not CHROMA_DIR.exists():
            return False

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(COLLECTION_NAME)

        if collection.count() != len(sections):
            return False

        # Verify that the vector index contains the expected chunk IDs.
        saved_ids = set(collection.get()["ids"])
        expected_ids = {section["id"] for section in sections}

        if saved_ids != expected_ids:
            return False

        return True

    except Exception as exc:
        print(f"Could not verify existing indexes: {exc}")
        return False


def build_vector_index(sections, model):
    print(f"Embedding {len(sections)} sections with {EMBED_MODEL_NAME} ...")

    texts = [index_text(section) for section in sections]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
    ).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[section["id"] for section in sections],
        embeddings=embeddings,
        documents=[section["text"] for section in sections],
        metadatas=[
            {
                "act": section["act"],
                "section": section["section"],
                "title": section["title"],
                "domain": section.get("domain", "general"),
            }
            for section in sections
        ],
    )

    print(f"Vector index built and saved to {CHROMA_DIR}")


def build_bm25_index(sections):
    print("Building BM25 keyword index ...")

    tokenized_corpus = [
        index_text(section).lower().split()
        for section in sections
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_FILE, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "sections": sections,
            },
            f,
        )

    print(f"BM25 index built and saved to {BM25_FILE}")


if __name__ == "__main__":
    print(f"Using data file: {DATA_FILE}")

    sections = load_sections()

    ids = [section["id"] for section in sections]

    if len(ids) != len(set(ids)):
        duplicates = sorted(
            {chunk_id for chunk_id in ids if ids.count(chunk_id) > 1}
        )[:10]

        raise SystemExit(
            f"Duplicate chunk IDs found: {duplicates}. "
            "Fix extract_acts.py first."
        )

    # Check before loading the embedding model.
    if indexes_are_current(sections):
        print(f"\nFound {len(sections)} matching chunks.")
        print("Existing Chroma and BM25 indexes are up to date.")
        print("Skipping embedding generation.")
        print("\nDone. Index is ready for querying.")
    else:
        print("\nIndexes are missing or out of date. Rebuilding...")

        model = SentenceTransformer(EMBED_MODEL_NAME)
        model.max_seq_length = MAX_SEQ_LENGTH

        build_vector_index(sections, model)
        build_bm25_index(sections)

        print("\nDone. Index is ready for querying.")