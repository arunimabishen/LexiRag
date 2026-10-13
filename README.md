# LexiRAG

Multilingual (English / Hindi / Hinglish), voice-ready RAG assistant for Indian legal questions, focused on Chhattisgarh. It answers **only** from the text of real Acts, cites the exact Act and Section, and **refuses and points to free legal aid** (NALSA / DLSA) when a question is risky, urgent or out of scope.

> This project gives general legal information, not legal advice.

## What it covers

| Prefix | Act | Domain |
| --- | --- | --- |
| CPA | Consumer Protection Act, 2019 | consumer |
| PWA | Payment of Wages Act, 1936 | labour |
| CGRCA | Chhattisgarh Rent Control Act, 2011 | tenancy |
| LSAA | Legal Services Authorities Act, 1987 | legal aid |

The two NALSA regulations PDFs are in `data/raw/` but are not indexed yet (poor scans, OCR needed).

## How it works

```
question -> language detection -> defer gate (keyword rules, then LLM scope check)
         -> Hindi/Hinglish rewritten to English for search
         -> hybrid retrieval (BM25 + bge-m3 vectors + fusion) -> cross-encoder re-rank
         -> grounded answer in the user's language (Groq) -> citations
```

Key files: `generation/pipeline.py` (`answer(query)`), `generation/gate.py`, `retrieval/hybrid_search.py`, `retrieval/rerank.py`, `retrieval/query_rewrite.py`, `api/main.py` (FastAPI).

## Setup (Windows, PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Groq API key: set it in YOUR terminal only. Never put it in a file or commit it.
$env:GROQ_API_KEY = "your-key-here"

# Build the search index (first run downloads BAAI/bge-m3, about 2 GB; takes ~5 minutes)
python ingestion\build_index.py
```

The first question also downloads the small re-ranker (about 0.5 GB). Both models are cached afterwards.

## Run the API

```powershell
python -m uvicorn api.main:app --port 8000
```

Wait for "Application startup complete" (about 30 seconds while models load), then open http://localhost:8000/docs to try it.

`POST /ask` with `{"query": "My landlord is not returning my deposit"}` returns JSON:

| Field | Meaning |
| --- | --- |
| `answer` | answer text, in the user's language |
| `language` | `en`, `hi` or `hinglish` |
| `deferred` | `true` when the question was refused (risky / out of scope) |
| `defer_reason`, `defer_reasons` | why it was deferred |
| `citations[]` | `marker`, `act`, `section`, `title`, `source_id`, `text`, `cited` |
| `resources[]` | helplines to show when deferred (NALSA 15100, DLSA, 112, Tele-MANAS) |
| `search_query`, `seconds` | the English search text used, and the time taken |

If `deferred` is `true`, the UI should show the legal-aid card instead of an answer. `GET /health` returns `{"status": "ok"}`.

## Evaluation

`eval/` holds the 51-question main set, a 28-question holdout and the scoring scripts. Retrieval numbers (small sets, scored at section level; read as encouraging, not proven):

| Set | n | hit@1 | hit@3 | hit@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| Main (answerable) | 44 | 0.68 | 0.91 | 0.91 | 0.81 |
| Holdout (answerable) | 14 | 0.71 | 0.93 | 0.93 | 0.81 |

The gate (rules + scope check) caught 21 of 21 defer questions with 0 wrong refusals on these sets.

```powershell
python eval\score_retrieval.py mytest rerank rewrite     # retrieval scores
python eval\check_sources.py                             # what retrieval returns for a few questions
python eval\score_scope.py                               # defer-gate check (needs GROQ key)
```

## Known limitations

- Multi-domain questions (for example wages and rent in one message) often retrieve only one law.
- Very short Hindi questions retrieve less precisely than fuller ones.
- Answers can occasionally misstate a detail or translate a legal term loosely; always shown with citations so they can be checked.
- The local Durg DLSA contact is not filled in yet; the generic DLSA line is shown.
- Evaluation sets are small and were written by the developers; answer-quality evaluation against raw LLMs is still to be done.

## Security

Keep API keys in environment variables only. `.env`, `venv/` and the generated index are git-ignored. If a key has ever been pasted into a chat or file, rotate it.