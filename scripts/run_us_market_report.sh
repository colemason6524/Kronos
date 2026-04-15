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
mkdir -p data "$REPORT_DIR"

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

REPORT_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_BASENAME="${SYMBOL,,}_report_${REPORT_STAMP}"
REPORT_PATH="${REPORT_DIR}/${REPORT_BASENAME}.txt"
REPORT_JSON_PATH="${REPORT_DIR}/${REPORT_BASENAME}.json"

echo
echo "Running Kronos live report..."
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
