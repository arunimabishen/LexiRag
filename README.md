# LexiRAG — Thin End-to-End Slice (Pathway C)

This proves the full pipeline shape works: **query → hybrid retrieval → grounded, cited answer.**
It uses a small sample of 8 illustrative law sections (`data/sample_sections.json`) —
not the real Acts yet. That's Pathway A, next.

## What's real vs. placeholder right now

| Component | Status |
|---|---|
| Chunking / section structure | ✅ Real logic |
| BM25 keyword search | ✅ Real, fully working |
| Hybrid fusion (RRF) | ✅ Real, fully working |
| Vector/semantic search | ⚠️ **Placeholder** — using TF-IDF instead of a real multilingual embedding model, because this sandbox can't reach huggingface.co to download one |
| LLM generation | ⚠️ **Placeholder** — template-based mock answer, because no API key is set in this sandbox |

Both placeholders are drop-in swaps once you run this on your own machine.

## Setup on your own machine

### 1. Install Python (if you don't have it)
Download Python 3.11+ from https://www.python.org/downloads/ (check "Add to PATH" during install on Windows).

### 2. Get the project files
Unzip the folder I'll give you, `cd` into it in a terminal.

### 3. Create a virtual environment (keeps dependencies isolated)
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 4. Install dependencies
```bash
pip install sentence-transformers chromadb rank_bm25 anthropic scikit-learn
```
This will download real packages AND the embedding model (~470MB, first run only)
from huggingface.co — this works fine on your machine since it's not network-restricted.

### 5. Swap TF-IDF back for the real multilingual embedding model
In `ingestion/build_index.py`, replace the `TfidfVectorizer` block with:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(texts).tolist()
```
And update `retrieval/hybrid_search.py`'s `vector_search()` to encode the query
with the same model instead of the TF-IDF vectorizer.

### 6. Get an LLM API key
Sign up at https://console.anthropic.com (or OpenAI/Google if you prefer) and
get an API key. Then set it as an environment variable:
```bash
export ANTHROPIC_API_KEY="your-key-here"     # Mac/Linux
setx ANTHROPIC_API_KEY "your-key-here"       # Windows
```
Once this is set, `generation/generate.py` will automatically use the real
`call_llm()` path instead of the mock.

### 7. Run it
```bash
python3 ingestion/build_index.py     # build the index (run once, or whenever data changes)
python3 generation/generate.py       # ask a test question end-to-end
```

## What to do next (Pathway A)
Replace `data/sample_sections.json` with real sections extracted from actual
Acts (Consumer Protection Act 2019, Model Tenancy Act, etc.) using
`pdfplumber`/`PyMuPDF` on the real PDF text from India Code.

## Files
```
lexirag/
├── data/sample_sections.json     # sample law sections (placeholder data)
├── ingestion/build_index.py      # chunking + embedding + index building
├── retrieval/hybrid_search.py    # BM25 + vector search + RRF fusion
└── generation/generate.py        # prompt building + LLM call (or mock)
```
