#!/usr/bin/env python3
"""
Generate a plain-text manual intel report from Kronos forecasts.

Two input modes are supported:
1. Analyze a saved Web UI prediction JSON.
2. Run a fresh Kronos forecast from a market CSV and summarize the output.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from model import Kronos, KronosPredictor, KronosTokenizer


DEFAULT_MODEL = "NeoQuasar/Kronos-small"
DEFAULT_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"


@dataclass
class ForecastBundle:
    instrument: str
    timeframe: str
    lookback: int
    pred_len: int
    context_df: pd.DataFrame
    pred_df: pd.DataFrame
    actual_df: Optional[pd.DataFrame]
    sample_count: int
    temperature: float
    top_p: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a text-based manual intel report from Kronos output."
    )
    parser.add_argument("--prediction-json", help="Path to a saved Web UI prediction JSON.")
    parser.add_argument("--csv", help="Path to a market CSV to forecast with Kronos.")
    parser.add_argument("--instrument", default="Unknown Instrument", help="Instrument label for the report.")
    parser.add_argument("--timeframe", default="", help="Timeframe label, e.g. 1h or 5m.")
    parser.add_argument("--lookback", type=int, default=400, help="Number of historical candles to use.")
    parser.add_argument("--pred-len", type=int, default=24, help="Number of candles to forecast.")
    parser.add_argument("--start-date", help="Optional start date for an in-sample comparison window.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Forecast beyond the end of the CSV instead of using the last pred_len rows as holdout actuals.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL, help="Hugging Face model id or local path.")
    parser.add_argument("--tokenizer-id", default=DEFAULT_TOKENIZER, help="Hugging Face tokenizer id or local path.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cpu, cuda:0, mps.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling probability.")
    parser.add_argument("--sample-count", type=int, default=3, help="Number of forecast samples to average.")
    parser.add_argument("--save-json", help="Optional path to save a machine-readable report payload.")
    parser.add_argument("--verbose", action="store_true", help="Show Kronos autoregressive progress.")
    args = parser.parse_args()

    if not args.prediction_json and not args.csv:
        parser.error("Provide either --prediction-json or --csv.")
    return args


def load_market_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "timestamps" in df.columns:
        ts_col = "timestamps"
    elif "timestamp" in df.columns:
        ts_col = "timestamp"
    elif "date" in df.columns:
        ts_col = "date"
    else:
        raise ValueError("CSV must include one of: timestamps, timestamp, date")

    df["timestamps"] = pd.to_datetime(df[ts_col])
    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    for col in required + [col for col in ["volume", "amount"] if col in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).sort_values("timestamps").reset_index(drop=True)
    return df


def infer_timeframe(df: pd.DataFrame) -> str:
    if len(df) < 2:
        return "unknown"

    deltas = df["timestamps"].diff().dropna()
    if deltas.empty:
        return "unknown"

    seconds = int(deltas.median().total_seconds())
    if seconds <= 0:
        return "unknown"
    if seconds % 86400 == 0:
        days = seconds // 86400
        return f"{days}d"
    if seconds % 3600 == 0:
        hours = seconds // 3600
        return f"{hours}h"
    if seconds % 60 == 0:
        minutes = seconds // 60
        return f"{minutes}m"
    return f"{seconds}s"


def future_timestamps(history: pd.Series, pred_len: int) -> pd.Series:
    if len(history) < 2:
        start = pd.Timestamp.utcnow().floor("h")
        freq = pd.Timedelta(hours=1)
    else:
        freq = history.diff().dropna().median()
        if pd.isna(freq) or freq <= pd.Timedelta(0):
            freq = pd.Timedelta(hours=1)
        start = history.iloc[-1] + freq
    return pd.Series(pd.date_range(start=start, periods=pred_len, freq=freq), name="timestamps")


def load_models(model_id: str, tokenizer_id: str, device: Optional[str]) -> KronosPredictor:
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
    model = Kronos.from_pretrained(model_id)
    tokenizer.eval()
    model.eval()
    return KronosPredictor(model, tokenizer, device=device, max_context=512)


def select_window(
    df: pd.DataFrame,
    lookback: int,
    pred_len: int,
    start_date: Optional[str],
    live: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, Optional[pd.DataFrame]]:
    if start_date:
        start_ts = pd.to_datetime(start_date)
        window_df = df[df["timestamps"] >= start_ts].reset_index(drop=True)
        if len(window_df) < lookback + pred_len:
            raise ValueError(
                f"Need at least {lookback + pred_len} rows after {start_ts}, found {len(window_df)}."
            )
        context_df = window_df.iloc[:lookback].copy()
        actual_df = window_df.iloc[lookback:lookback + pred_len].copy()
        y_timestamp = actual_df["timestamps"].reset_index(drop=True)
        return context_df, context_df["timestamps"].reset_index(drop=True), y_timestamp, actual_df

    if live:
        if len(df) < lookback:
            raise ValueError(f"Need at least {lookback} rows, found {len(df)}.")

        context_df = df.iloc[-lookback:].copy()
        y_timestamp = future_timestamps(context_df["timestamps"].reset_index(drop=True), pred_len)
        return context_df, context_df["timestamps"].reset_index(drop=True), y_timestamp, None

    if len(df) >= lookback + pred_len:
        context_df = df.iloc[-(lookback + pred_len):-pred_len].copy()
        actual_df = df.iloc[-pred_len:].copy()
        y_timestamp = actual_df["timestamps"].reset_index(drop=True)
        return context_df, context_df["timestamps"].reset_index(drop=True), y_timestamp, actual_df

    if len(df) < lookback:
        raise ValueError(f"Need at least {lookback} rows, found {len(df)}.")

    context_df = df.iloc[-lookback:].copy()
    y_timestamp = future_timestamps(context_df["timestamps"].reset_index(drop=True), pred_len)
    return context_df, context_df["timestamps"].reset_index(drop=True), y_timestamp, None


def bundle_from_csv(args: argparse.Namespace) -> ForecastBundle:
    df = load_market_csv(args.csv)
    timeframe = args.timeframe or infer_timeframe(df)
    context_df, x_timestamp, y_timestamp, actual_df = select_window(
        df,
        args.lookback,
        args.pred_len,
        args.start_date,
        args.live,
    )

    feature_cols = ["open", "high", "low", "close"]
    if "volume" in context_df.columns:
        feature_cols.append("volume")
    if "amount" in context_df.columns:
        feature_cols.append("amount")

    predictor = load_models(args.model_id, args.tokenizer_id, args.device)
    pred_df = predictor.predict(
        df=context_df[feature_cols].copy(),
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=args.pred_len,
        T=args.temperature,
        top_p=args.top_p,
        sample_count=args.sample_count,
        verbose=args.verbose,
    ).reset_index().rename(columns={"index": "timestamps"})

    if "timestamps" not in pred_df.columns:
        pred_df["timestamps"] = y_timestamp

    return ForecastBundle(
        instrument=args.instrument,
        timeframe=timeframe,
        lookback=args.lookback,
        pred_len=args.pred_len,
        context_df=context_df.reset_index(drop=True),
        pred_df=pred_df.reset_index(drop=True),
        actual_df=None if actual_df is None else actual_df.reset_index(drop=True),
        sample_count=args.sample_count,
        temperature=args.temperature,
        top_p=args.top_p,
    )


def bundle_from_prediction_json(path: str, instrument: str, timeframe: str) -> ForecastBundle:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    prediction_results = pd.DataFrame(raw["prediction_results"])
    if prediction_results.empty:
        raise ValueError("Prediction JSON does not contain prediction_results.")

    prediction_results["timestamps"] = pd.to_datetime(prediction_results["timestamp"])
    prediction_results = prediction_results.drop(columns=["timestamp"])

    actual_df = None
    if "actual_data" in raw and raw["actual_data"]:
        actual_df = pd.DataFrame(raw["actual_data"])
        if not actual_df.empty:
            actual_df["timestamps"] = pd.to_datetime(actual_df["timestamp"])
            actual_df = actual_df.drop(columns=["timestamp"])

    context_df = pd.DataFrame()
    lookback = int(raw.get("prediction_params", {}).get("lookback", 0))
    pred_len = int(raw.get("prediction_params", {}).get("pred_len", len(prediction_results)))
    sample_count = int(raw.get("prediction_params", {}).get("sample_count", 1))
    temperature = float(raw.get("prediction_params", {}).get("temperature", 1.0))
    top_p = float(raw.get("prediction_params", {}).get("top_p", 0.9))

    inferred_timeframe = timeframe or "unknown"
    if actual_df is not None and len(actual_df) > 1:
        inferred_timeframe = inferred_timeframe or infer_timeframe(actual_df)

    return ForecastBundle(
        instrument=instrument,
        timeframe=inferred_timeframe,
        lookback=lookback,
        pred_len=pred_len,
        context_df=context_df,
        pred_df=prediction_results.reset_index(drop=True),
        actual_df=actual_df.reset_index(drop=True) if actual_df is not None else None,
        sample_count=sample_count,
        temperature=temperature,
        top_p=top_p,
    )


def pct_change(new_value: float, base_value: float) -> float:
    if base_value == 0:
        return math.nan
    return ((new_value - base_value) / base_value) * 100.0


def describe_trend(closes: pd.Series) -> str:
    if len(closes) < 10:
        return "insufficient context"

    recent = closes.tail(min(48, len(closes))).to_numpy(dtype=float)
    start = recent[0]
    end = recent[-1]
    move_pct = pct_change(end, start)
    if abs(move_pct) < 0.4:
        return "sideways"
    return "uptrend" if move_pct > 0 else "downtrend"


def describe_market_condition(context_df: pd.DataFrame) -> str:
    if context_df.empty or len(context_df) < 20:
        return "unknown"

    closes = context_df["close"].to_numpy(dtype=float)
    highs = context_df["high"].to_numpy(dtype=float)
    lows = context_df["low"].to_numpy(dtype=float)
    total_move = abs(pct_change(closes[-1], closes[0]))
    avg_range_pct = float(np.mean((highs - lows) / np.maximum(closes, 1e-9)) * 100.0)

    if total_move >= 3 and avg_range_pct >= 1.2:
        return "volatile trend"
    if total_move >= 2:
        return "trending"
    if avg_range_pct >= 1.5:
        return "volatile range"
    return "compressed/ranging"


def describe_bias(final_return_pct: float) -> str:
    if final_return_pct > 0.35:
        return "bullish"
    if final_return_pct < -0.35:
        return "bearish"
    return "neutral"


def describe_path_shape(pred_close: pd.Series, current_close: float) -> str:
    close_arr = pred_close.to_numpy(dtype=float)
    final_return = pct_change(close_arr[-1], current_close)
    net_move = close_arr[-1] - close_arr[0]
    diff_signs = np.sign(np.diff(close_arr))
    sign_changes = int(np.sum(diff_signs[1:] != diff_signs[:-1])) if len(diff_signs) > 1 else 0

    if abs(final_return) < 0.25 and sign_changes >= max(2, len(close_arr) // 5):
        return "range/chop"
    if abs(final_return) >= 0.6 and sign_changes <= max(1, len(close_arr) // 8):
        return "trend continuation" if net_move != 0 else "range/chop"
    if close_arr.max() > current_close and close_arr.min() < current_close:
        return "two-sided / choppy"
    return "gradual drift"


def expected_pullback(pred_df: pd.DataFrame, current_close: float, bias: str) -> str:
    if bias == "bullish":
        drawdown = pct_change(pred_df["low"].min(), current_close)
        return "yes" if drawdown < -0.2 else "minor"
    if bias == "bearish":
        squeeze = pct_change(pred_df["high"].max(), current_close)
        return "yes" if squeeze > 0.2 else "minor"
    return "not directional enough"


def expected_volatility(pred_df: pd.DataFrame, current_close: float) -> str:
    avg_range_pct = float(np.mean((pred_df["high"] - pred_df["low"]) / np.maximum(pred_df["close"], 1e-9)) * 100.0)
    total_span_pct = max(
        abs(pct_change(float(pred_df["high"].max()), current_close)),
        abs(pct_change(float(pred_df["low"].min()), current_close)),
    )
    score = max(avg_range_pct, total_span_pct)
    if score >= 2.5:
        return "high"
    if score >= 1.0:
        return "medium"
    return "low"


def agreement_label(sample_count: int) -> str:
    if sample_count <= 1:
        return "not measured (single path)"
    if sample_count <= 3:
        return "medium"
    return "medium by default; inspect raw paths if you add per-sample export"


def confidence_note(bias: str, path_shape: str, sample_count: int) -> str:
    if sample_count <= 1:
        base = "single averaged path only, so confidence is limited"
    else:
        base = "multiple samples were averaged, which helps smooth noise but hides disagreement"

    if bias == "neutral":
        return f"{base}; best used as a no-trade or low-conviction read."
    if "chop" in path_shape:
        return f"{base}; directional conviction looks weak because the path is choppy."
    return f"{base}; useful for directional framing, but not precise execution."


def actual_outcome_section(bundle: ForecastBundle) -> list[str]:
    if bundle.actual_df is None or bundle.actual_df.empty:
        return [
            "- What actually happened: not available in this run",
            "- Was direction correct: n/a",
            "- Was volatility estimated well: n/a",
            "- Did the model miss a regime shift: n/a",
            "- Lesson for next run: compare this forecast against future realized candles",
        ]

    pred_final = float(bundle.pred_df["close"].iloc[-1])
    actual_final = float(bundle.actual_df["close"].iloc[-1])
    current_close = float(bundle.actual_df["close"].iloc[0])
    pred_dir = np.sign(pred_final - current_close)
    actual_dir = np.sign(actual_final - current_close)
    direction_correct = "yes" if pred_dir == actual_dir else "no"

    pred_range = float(bundle.pred_df["high"].max() - bundle.pred_df["low"].min())
    actual_range = float(bundle.actual_df["high"].max() - bundle.actual_df["low"].min())
    range_diff_pct = pct_change(actual_range, pred_range) if pred_range else math.nan

    if math.isnan(range_diff_pct):
        vol_read = "n/a"
    elif abs(range_diff_pct) <= 20:
        vol_read = "roughly yes"
    elif range_diff_pct > 20:
        vol_read = "model underestimated realized volatility"
    else:
        vol_read = "model overestimated realized volatility"

    lesson = "good candidate to archive and compare with future runs"
    if direction_correct == "no":
        lesson = "check whether the context window was regime-mismatched or the market was event-driven"

    return [
        f"- What actually happened: final realized close {actual_final:.4f} vs predicted {pred_final:.4f}",
        f"- Was direction correct: {direction_correct}",
        f"- Was volatility estimated well: {vol_read}",
        "- Did the model miss a regime shift: review alongside news/session context",
        f"- Lesson for next run: {lesson}",
    ]


def records_with_iso_timestamps(df: Optional[pd.DataFrame]) -> list[dict]:
    if df is None or df.empty:
        return []

    safe_df = df.copy()
    if "timestamps" in safe_df.columns:
        safe_df["timestamps"] = safe_df["timestamps"].apply(lambda ts: pd.Timestamp(ts).isoformat())
    return safe_df.to_dict(orient="records")


def summarize_bundle(bundle: ForecastBundle) -> dict:
    current_close = float(bundle.context_df["close"].iloc[-1]) if not bundle.context_df.empty else float(bundle.pred_df["close"].iloc[0])
    final_close = float(bundle.pred_df["close"].iloc[-1])
    highest_high = float(bundle.pred_df["high"].max())
    lowest_low = float(bundle.pred_df["low"].min())

    final_return_pct = pct_change(final_close, current_close)
    max_upside_pct = pct_change(highest_high, current_close)
    max_downside_pct = pct_change(lowest_low, current_close)

    bias = describe_bias(final_return_pct)
    path_shape = describe_path_shape(bundle.pred_df["close"], current_close)
    trend_impression = describe_trend(bundle.context_df["close"]) if not bundle.context_df.empty else "not available"
    market_condition = describe_market_condition(bundle.context_df) if not bundle.context_df.empty else "not available"
    pullback = expected_pullback(bundle.pred_df, current_close, bias)
    volatility = expected_volatility(bundle.pred_df, current_close)
    agreement = agreement_label(bundle.sample_count)
    confidence = confidence_note(bias, path_shape, bundle.sample_count)

    if bias == "bullish":
        bull_case = f"path extends toward {final_close:.4f} to {highest_high:.4f}"
        bear_case = f"loss of structure below {lowest_low:.4f} weakens the setup"
        zone = f"{final_close:.4f} to {highest_high:.4f}"
        invalidation = f"below {lowest_low:.4f}"
        watch_for = "continuation after a shallow pullback that preserves structure"
        avoid_if = "price rejects the forecast path early and breaks the downside zone"
    elif bias == "bearish":
        bull_case = f"squeeze above {highest_high:.4f} would challenge the short thesis"
        bear_case = f"path extends toward {final_close:.4f} with weakness down to {lowest_low:.4f}"
        zone = f"{lowest_low:.4f} to {final_close:.4f}"
        invalidation = f"above {highest_high:.4f}"
        watch_for = "failed rallies that keep the path pointed lower"
        avoid_if = "price reclaims the upside zone and invalidates the downside path"
    else:
        bull_case = "no strong directional edge"
        bear_case = "no strong directional edge"
        zone = f"{lowest_low:.4f} to {highest_high:.4f}"
        invalidation = "not directional enough to define tightly"
        watch_for = "whether price resolves out of the predicted range cleanly"
        avoid_if = "you need strong directional conviction"

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "instrument": bundle.instrument,
        "timeframe": bundle.timeframe or "unknown",
        "lookback": bundle.lookback,
        "pred_len": bundle.pred_len,
        "sample_count": bundle.sample_count,
        "temperature": bundle.temperature,
        "top_p": bundle.top_p,
        "context": {
            "current_close": current_close,
            "trend_impression": trend_impression,
            "market_condition": market_condition,
            "context_end_timestamp": None if bundle.context_df.empty else pd.Timestamp(bundle.context_df["timestamps"].iloc[-1]).isoformat(),
        },
        "forecast": {
            "predicted_final_close": final_close,
            "predicted_return_pct": final_return_pct,
            "highest_predicted_high": highest_high,
            "lowest_predicted_low": lowest_low,
            "max_upside_pct": max_upside_pct,
            "max_downside_pct": max_downside_pct,
            "bias": bias,
            "path_shape": path_shape,
            "expected_pullback": pullback,
            "expected_volatility": volatility,
            "agreement": agreement,
            "confidence_note": confidence,
            "bull_case": bull_case,
            "bear_case": bear_case,
            "most_likely_zone": zone,
            "invalidation_zone": invalidation,
            "watch_for": watch_for,
            "avoid_if": avoid_if,
        },
        "forecast_timestamps": [pd.Timestamp(ts).isoformat() for ts in bundle.pred_df["timestamps"]],
        "forecast_rows": records_with_iso_timestamps(bundle.pred_df),
        "actual_rows": records_with_iso_timestamps(bundle.actual_df),
    }


def format_report(bundle: ForecastBundle) -> str:
    summary = summarize_bundle(bundle)
    current_close = summary["context"]["current_close"]
    final_close = summary["forecast"]["predicted_final_close"]
    highest_high = summary["forecast"]["highest_predicted_high"]
    lowest_low = summary["forecast"]["lowest_predicted_low"]
    final_return_pct = summary["forecast"]["predicted_return_pct"]
    max_upside_pct = summary["forecast"]["max_upside_pct"]
    max_downside_pct = summary["forecast"]["max_downside_pct"]
    bias = summary["forecast"]["bias"]
    path_shape = summary["forecast"]["path_shape"]
    trend_impression = summary["context"]["trend_impression"]
    market_condition = summary["context"]["market_condition"]
    pullback = summary["forecast"]["expected_pullback"]
    volatility = summary["forecast"]["expected_volatility"]
    agreement = summary["forecast"]["agreement"]
    confidence = summary["forecast"]["confidence_note"]
    bull_case = summary["forecast"]["bull_case"]
    bear_case = summary["forecast"]["bear_case"]
    zone = summary["forecast"]["most_likely_zone"]
    invalidation = summary["forecast"]["invalidation_zone"]
    watch_for = summary["forecast"]["watch_for"]
    avoid_if = summary["forecast"]["avoid_if"]

    lines = [
        f"Instrument: {bundle.instrument}",
        f"Timeframe: {bundle.timeframe or 'unknown'}",
        f"Context Window: {bundle.lookback} candles",
        f"Forecast Horizon: {bundle.pred_len} candles",
        "",
        "Current Market",
        f"- Current close: {current_close:.4f}",
        f"- Current trend impression: {trend_impression}",
        f"- Market condition before forecast: {market_condition}",
        "",
        "Forecast Summary",
        f"- Predicted final close: {final_close:.4f}",
        f"- Predicted return to final close: {final_return_pct:+.2f}%",
        f"- Highest predicted high: {highest_high:.4f}",
        f"- Lowest predicted low: {lowest_low:.4f}",
        f"- Max upside from current price: {max_upside_pct:+.2f}%",
        f"- Max downside from current price: {max_downside_pct:+.2f}%",
        "",
        "Path Read",
        f"- Bias: {bias}",
        f"- Path shape: {path_shape}",
        f"- Expected pullback before move: {pullback}",
        f"- Expected volatility: {volatility}",
        "",
        "Scenario Quality",
        f"- Sample count: {bundle.sample_count}",
        f"- Agreement across samples: {agreement}",
        "- Bullish paths vs bearish paths: not exported separately by the base repo",
        f"- Confidence note: {confidence}",
        "",
        "Trade Interpretation",
        f"- Bull case: {bull_case}",
        f"- Bear case: {bear_case}",
        f"- Most likely zone: {zone}",
        f"- Invalidation zone: {invalidation}",
        "- Conditions that would make this setup unattractive: news shock, regime change, or high event risk",
        "",
        "Manual Action Idea",
        f"- Watch for: {watch_for}",
        f"- Avoid if: {avoid_if}",
        "- Best use of forecast: directional bias, target planning, and no-trade filtering",
        "",
        "Post-Result Review",
    ]
    lines.extend(actual_outcome_section(bundle))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.prediction_json:
        bundle = bundle_from_prediction_json(args.prediction_json, args.instrument, args.timeframe)
    else:
        bundle = bundle_from_csv(args)
    report_text = format_report(bundle)
    print(report_text)
    if args.save_json:
        payload = summarize_bundle(bundle)
        output_dir = os.path.dirname(args.save_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    main()
