#!/usr/bin/env python3
"""
Review a saved live Kronos report against realized market candles.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from scripts.kronos_text_report import load_market_csv, pct_change


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a saved Kronos report JSON against realized candles.")
    parser.add_argument("--report-json", required=True, help="Path to a saved machine-readable report JSON.")
    parser.add_argument("--csv", required=True, help="Path to a refreshed market CSV containing realized candles.")
    return parser.parse_args()


def load_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def direction_label(delta: float) -> str:
    if delta > 0:
        return "up"
    if delta < 0:
        return "down"
    return "flat"


def volatility_read(pred_range: float, actual_range: float) -> str:
    if pred_range <= 0:
        return "n/a"
    diff_pct = pct_change(actual_range, pred_range)
    if math.isnan(diff_pct):
        return "n/a"
    if abs(diff_pct) <= 20:
        return "roughly matched"
    if diff_pct > 20:
        return "underestimated realized volatility"
    return "overestimated realized volatility"


def main() -> None:
    args = parse_args()
    report = load_report(args.report_json)
    df = load_market_csv(args.csv)

    forecast_timestamps = pd.to_datetime(report["forecast_timestamps"])
    realized = df[df["timestamps"].isin(forecast_timestamps)].copy().sort_values("timestamps").reset_index(drop=True)

    if realized.empty:
        raise ValueError("No realized candles from the report horizon were found in the supplied CSV.")

    expected_len = len(forecast_timestamps)
    available_len = len(realized)

    pred_df = pd.DataFrame(report["forecast_rows"]).copy()
    pred_df["timestamps"] = pd.to_datetime(pred_df["timestamps"])
    pred_df = pred_df[pred_df["timestamps"].isin(realized["timestamps"])].sort_values("timestamps").reset_index(drop=True)

    pred_current_close = float(report["context"]["current_close"])
    pred_final = float(pred_df["close"].iloc[-1])
    actual_final = float(realized["close"].iloc[-1])
    pred_return = pct_change(pred_final, pred_current_close)
    actual_return = pct_change(actual_final, pred_current_close)
    pred_dir = direction_label(pred_final - pred_current_close)
    actual_dir = direction_label(actual_final - pred_current_close)
    direction_correct = "yes" if pred_dir == actual_dir else "no"

    pred_high = float(pred_df["high"].max())
    pred_low = float(pred_df["low"].min())
    actual_high = float(realized["high"].max())
    actual_low = float(realized["low"].min())
    pred_range = pred_high - pred_low
    actual_range = actual_high - actual_low

    print(f"Instrument: {report['instrument']}")
    print(f"Timeframe: {report['timeframe']}")
    print(f"Generated at: {report['generated_at_utc']}")
    print(f"Forecast candles expected: {expected_len}")
    print(f"Forecast candles available in realized data: {available_len}")
    print()
    print("Forecast vs Realized")
    print(f"- Starting close used by report: {pred_current_close:.4f}")
    print(f"- Predicted final close: {pred_final:.4f} ({pred_return:+.2f}%)")
    print(f"- Realized final close: {actual_final:.4f} ({actual_return:+.2f}%)")
    print(f"- Predicted direction: {pred_dir}")
    print(f"- Realized direction: {actual_dir}")
    print(f"- Was direction correct: {direction_correct}")
    print()
    print("Range Review")
    print(f"- Predicted high/low: {pred_high:.4f} / {pred_low:.4f}")
    print(f"- Realized high/low: {actual_high:.4f} / {actual_low:.4f}")
    print(f"- Volatility read: {volatility_read(pred_range, actual_range)}")
    print()
    print("Takeaway")
    if available_len < expected_len:
        print("- Realized data is incomplete for the full horizon, so treat this review as partial.")
    elif direction_correct == "yes":
        print("- The directional call was correct. Compare the timing and volatility next.")
    else:
        print("- The directional call missed. Review whether the market regime changed or the setup was too noisy.")


if __name__ == "__main__":
    main()
