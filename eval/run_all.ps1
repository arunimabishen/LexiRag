# One command for the whole check. Run from the project root:   .\eval\run_all.ps1
# Takes roughly 25-35 minutes (re-extract, re-index, four scoring runs). Output is also saved
# to eval\run_all_output.txt.
$env:RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
$env:RERANK_POOL = "20"

& {
    python ingestion\extract_acts.py
    python ingestion\build_index.py
    python eval\score_retrieval.py clean_rerank rerank
    python eval\score_retrieval.py rewrite rerank rewrite
    python eval\score_retrieval.py holdout_clean rerank eval\gate_holdout.json
    python eval\score_retrieval.py holdout_rewrite rerank eval\gate_holdout.json rewrite
    python eval\compare_runs.py
} *>&1 | Tee-Object -FilePath eval\run_all_output.txt