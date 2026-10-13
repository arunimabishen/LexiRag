# Re-runs only the scoring (no re-extract, no re-index). Run from the project root:
#   .\eval\run_scoring.ps1
# Takes roughly 15-20 minutes. Output is also saved to eval\run_scoring_output.txt.
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
$env:RERANK_POOL = "20"

& {
    python eval\score_retrieval.py clean_rerank rerank
    python eval\score_retrieval.py rewrite rerank rewrite
    python eval\score_retrieval.py holdout_clean rerank eval\gate_holdout.json
    python eval\compare_runs.py
} *>&1 | Tee-Object -FilePath eval\run_scoring_output.txt