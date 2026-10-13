# Measures domain routing and the deposit-bonus ablation. Run from the project root:
#   .\eval\run_improve.ps1
# Takes roughly 25 minutes. Output also saved to eval\run_improve_output.txt.
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
$env:RERANK_POOL = "20"

& {
    python eval\score_retrieval.py route rerank rewrite route
    python eval\score_retrieval.py holdout_route rerank eval\gate_holdout.json rewrite route
    $env:LEXIRAG_NO_DEPOSIT_BONUS = "1"
    python eval\score_retrieval.py nohack rerank rewrite
    Remove-Item Env:LEXIRAG_NO_DEPOSIT_BONUS
    python eval\compare_runs.py
} *>&1 | Tee-Object -FilePath eval\run_improve_output.txt