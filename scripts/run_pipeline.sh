#!/usr/bin/env bash
# Run the local pipeline notebook headless (survives closing the editor / terminal).
#   bash scripts/run_pipeline.sh dry    # tiny end-to-end test  -> outputs_dryrun/
#   bash scripts/run_pipeline.sh full   # full experiments      -> outputs/
# Re-running after a crash resumes: finished stages are skipped, training resumes from <run>/last/.
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/ML_Research/.venv/bin/activate
MODE="${1:-dry}"
case "$MODE" in
  dry) DRY=True ;;
  full) DRY=False ;;
  *) echo "usage: $0 [dry|full]"; exit 1 ;;
esac
mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/run_${MODE}_${TS}.log"
nohup papermill notebooks/PA2_local_pipeline.ipynb "logs/executed_${MODE}_${TS}.ipynb" \
  -k ml -p DRY_RUN "$DRY" --log-output --request-save-on-cell-execute > "$LOG" 2>&1 &
echo "started PID $! (mode=$MODE)"
echo "follow progress:  tail -f $LOG"
echo "executed notebook (saved after every cell): logs/executed_${MODE}_${TS}.ipynb"
