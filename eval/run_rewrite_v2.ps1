# Measures the improved English rewrite (adds the Acts' own vocabulary). Run from the project root:
#   .\eval\run_rewrite_v2.ps1
# Takes roughly 15-20 minutes. Output also saved to eval\run_rewrite_v2_output.txt.
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
$env:RERANK_POOL = "20"

& {
    python eval\score_retrieval.py rewrite_v2 rerank rewrite
    python eval\score_retrieval.py holdout_rewrite_v2 rerank eval\gate_holdout.json rewrite
    python eval\compare_runs.py
} *>&1 | Tee-Object -FilePath eval\run_rewrite_v2_output.txt