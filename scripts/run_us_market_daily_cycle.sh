#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv in $ROOT_DIR"
  echo "Create it first with: python3 -m venv .venv"
  exit 1
fi

source .venv/bin/activate

SYMBOL="${SYMBOL:-QQQ}"
INTERVAL="${INTERVAL:-5m}"
PERIOD="${PERIOD:-60d}"
LOOKBACK="${LOOKBACK:-400}"
PRED_LEN="${PRED_LEN:-78}"
SAMPLE_COUNT="${SAMPLE_COUNT:-3}"
INSTRUMENT_LABEL="${INSTRUMENT_LABEL:-$SYMBOL}"
PREPOST="${PREPOST:-false}"

CSV_PATH="data/${SYMBOL,,}_${INTERVAL}_latest.csv"
REPORT_DIR="reports"
LOG_DIR="logs"
REPORT_PREFIX="${SYMBOL,,}_report_"
mkdir -p data "$REPORT_DIR" "$LOG_DIR"

FETCH_ARGS=(
  --symbol "$SYMBOL"
  --interval "$INTERVAL"
  --period "$PERIOD"
  --output "$CSV_PATH"
)

if [[ "$PREPOST" == "true" ]]; then
  FETCH_ARGS+=(--prepost)
fi

echo "Refreshing ${INSTRUMENT_LABEL} data..."
python scripts/fetch_yfinance_klines.py "${FETCH_ARGS[@]}"

LATEST_CSV_TS="$(
python - "$CSV_PATH" <<'PY'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1], parse_dates=["timestamps"])
if df.empty:
    raise SystemExit("CSV has no rows.")
print(pd.Timestamp(df["timestamps"].max()).isoformat())
PY
)"

REVIEW_JSON="$(
python - "$REPORT_DIR" "$REPORT_PREFIX" "$LATEST_CSV_TS" <<'PY'
import glob
import json
import os
import sys
import pandas as pd

report_dir = sys.argv[1]
report_prefix = sys.argv[2]
latest_csv_ts = pd.Timestamp(sys.argv[3])
candidates = sorted(glob.glob(os.path.join(report_dir, f"{report_prefix}*.json")), reverse=True)

for path in candidates:
    if path.endswith("_review.json"):
        continue
    review_txt = path[:-5] + "_review.txt"
    if os.path.exists(review_txt):
        continue
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (json.JSONDecodeError, OSError):
        continue
    forecast_timestamps = payload.get("forecast_timestamps", [])
    if not forecast_timestamps:
        continue
    forecast_end = pd.Timestamp(forecast_timestamps[-1])
    if forecast_end <= latest_csv_ts:
        print(path)
        break
PY
)"

if [[ -n "${REVIEW_JSON:-}" ]]; then
  REVIEW_TXT="${REVIEW_JSON%.json}_review.txt"
  echo
  echo "Reviewing prior report: ${REVIEW_JSON}"
  python scripts/review_kronos_report.py \
    --report-json "$REVIEW_JSON" \
    --csv "$CSV_PATH" | tee "$REVIEW_TXT"
  echo "Saved review to: ${ROOT_DIR}/${REVIEW_TXT}"
else
  echo
  echo "No fully reviewable prior ${INSTRUMENT_LABEL} report found yet."
fi

REPORT_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_PATH="${REPORT_DIR}/${SYMBOL,,}_report_${REPORT_STAMP}.txt"
REPORT_JSON_PATH="${REPORT_DIR}/${SYMBOL,,}_report_${REPORT_STAMP}.json"

echo
echo "Running new Kronos live report..."
python scripts/kronos_text_report.py \
  --csv "$CSV_PATH" \
  --instrument "$INSTRUMENT_LABEL" \
  --timeframe "$INTERVAL" \
  --market-calendar us_equities \
  --live \
  --lookback "$LOOKBACK" \
  --pred-len "$PRED_LEN" \
  --sample-count "$SAMPLE_COUNT" \
  --save-json "$REPORT_JSON_PATH" | tee "$REPORT_PATH"

echo
echo "Saved latest CSV to: ${ROOT_DIR}/${CSV_PATH}"
echo "Saved report log to: ${ROOT_DIR}/${REPORT_PATH}"
echo "Saved report JSON to: ${ROOT_DIR}/${REPORT_JSON_PATH}"
