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

START_DATE="${START_DATE:-2026-03-01}"
END_DATE="${END_DATE:-$(date -u +%F)}"
INTERVAL="${INTERVAL:-1h}"
LOOKBACK="${LOOKBACK:-400}"
PRED_LEN="${PRED_LEN:-24}"
SAMPLE_COUNT="${SAMPLE_COUNT:-3}"
SYMBOL="${SYMBOL:-BTCUSDT}"
INSTRUMENT_LABEL="${INSTRUMENT_LABEL:-BTC/USDT}"

CSV_PATH="data/${SYMBOL,,}_${INTERVAL}_latest.csv"
REPORT_DIR="reports"
mkdir -p data "$REPORT_DIR"

echo "Refreshing ${INSTRUMENT_LABEL} data through ${END_DATE} UTC..."
python scripts/fetch_binance_vision_klines.py \
  --symbol "$SYMBOL" \
  --interval "$INTERVAL" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --output "$CSV_PATH" \
  --force

REPORT_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_PATH="${REPORT_DIR}/btc_report_${REPORT_STAMP}.txt"

echo
echo "Running Kronos live report..."
python scripts/kronos_text_report.py \
  --csv "$CSV_PATH" \
  --instrument "$INSTRUMENT_LABEL" \
  --timeframe "$INTERVAL" \
  --live \
  --lookback "$LOOKBACK" \
  --pred-len "$PRED_LEN" \
  --sample-count "$SAMPLE_COUNT" | tee "$REPORT_PATH"

echo
echo "Saved latest CSV to: ${ROOT_DIR}/${CSV_PATH}"
echo "Saved report log to: ${ROOT_DIR}/${REPORT_PATH}"
