#!/usr/bin/env python3
"""
Download Yahoo Finance OHLCV candles and convert them to a Kronos-ready CSV.

This is intended for U.S. stocks and ETFs such as QQQ, SPY, AAPL, and NVDA.
"""

from __future__ import annotations

import argparse
import os
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


EASTERN_TZ = ZoneInfo("America/New_York")
SUPPORTED_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m",
    "1h", "1d", "5d", "1wk", "1mo", "3mo",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Yahoo Finance candles and save them in Kronos CSV format."
    )
    parser.add_argument("--symbol", default="QQQ", help="Ticker symbol, e.g. QQQ or AAPL.")
    parser.add_argument("--interval", default="5m", help="Yahoo interval, e.g. 5m or 1d.")
    parser.add_argument(
        "--period",
        default="60d",
        help="Yahoo lookback period, e.g. 5d, 1mo, 60d, 1y. Ignored if --start-date is supplied.",
    )
    parser.add_argument("--start-date", help="Inclusive local date, format YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Exclusive local date, format YYYY-MM-DD.")
    parser.add_argument(
        "--output",
        help="Output CSV path. Defaults to data/<symbol>_<interval>_latest.csv under the repo.",
    )
    parser.add_argument(
        "--prepost",
        action="store_true",
        help="Include pre-market and after-hours candles.",
    )
    return parser.parse_args()


def kronos_output_path(symbol: str, interval: str) -> str:
    return os.path.join("data", f"{symbol.lower()}_{interval}_latest.csv")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamps", "open", "high", "low", "close", "volume", "amount"])

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            col[0] if isinstance(col, tuple) and len(col) > 0 and col[0] else col[-1]
            for col in out.columns
        ]

    out = out.reset_index()
    ts_col = "Datetime" if "Datetime" in out.columns else "Date"
    out["timestamps"] = pd.to_datetime(out[ts_col], errors="coerce")
    out = out.dropna(subset=["timestamps"])
    if getattr(out["timestamps"].dt, "tz", None) is not None:
        out["timestamps"] = out["timestamps"].dt.tz_convert(EASTERN_TZ).dt.tz_localize(None)

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    out = out.rename(columns=rename_map)

    required = ["open", "high", "low", "close"]
    for col in required + ["volume"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=required)
    out["volume"] = out["volume"].fillna(0.0)
    out["amount"] = out["close"] * out["volume"]

    keep_cols = ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    out = out[keep_cols].sort_values("timestamps").drop_duplicates(subset=["timestamps"]).reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    symbol = args.symbol.upper()
    interval = args.interval
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval {interval}. Choose from: {sorted(SUPPORTED_INTERVALS)}")

    output_path = args.output or kronos_output_path(symbol, interval)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(output_path):
        output_path = os.path.join(root_dir, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    download_kwargs = {
        "tickers": symbol,
        "interval": interval,
        "auto_adjust": False,
        "progress": False,
        "threads": False,
        "prepost": args.prepost,
    }
    if args.start_date:
        download_kwargs["start"] = args.start_date
        if args.end_date:
            download_kwargs["end"] = args.end_date
    else:
        download_kwargs["period"] = args.period

    print(f"Downloading {symbol} candles from Yahoo Finance...")
    raw_df = yf.download(**download_kwargs)
    if raw_df.empty:
        raise ValueError("Yahoo Finance returned no rows for this request.")

    out_df = normalize_columns(raw_df)
    if out_df.empty:
        raise ValueError("No usable OHLCV rows remained after normalization.")

    out_df.to_csv(output_path, index=False)
    print(f"Saved {len(out_df)} rows to {output_path}")
    print(f"Range: {out_df['timestamps'].min()} -> {out_df['timestamps'].max()}")


if __name__ == "__main__":
    main()
