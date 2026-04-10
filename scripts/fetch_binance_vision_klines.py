#!/usr/bin/env python3
"""
Download Binance public kline archives and convert them to a Kronos-ready CSV.

This uses Binance's public historical file archive rather than authenticated API keys.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import io
import os
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import pandas as pd


BASE_URL = "https://data.binance.vision/data/spot"
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]
SUPPORTED_INTERVALS = {
    "1s", "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1mo",
}


@dataclass(frozen=True)
class ArchiveTarget:
    period_type: str
    key: str
    url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Binance public klines and save them in Kronos CSV format."
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Spot symbol, e.g. BTCUSDT.")
    parser.add_argument("--interval", default="1h", help="Kline interval, e.g. 1h or 5m.")
    parser.add_argument("--start-date", required=True, help="Inclusive UTC date, format YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Inclusive UTC date, format YYYY-MM-DD. Defaults to today UTC.")
    parser.add_argument(
        "--output",
        help="Output CSV path. Defaults to data/<symbol>_<interval>_<start>_<end>.csv under the repo.",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/binance_vision",
        help="Directory for downloaded zip archives.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download archives even if they are already cached.",
    )
    return parser.parse_args()


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def month_floor(d: date) -> date:
    return d.replace(day=1)


def next_month(d: date) -> date:
    year = d.year + (1 if d.month == 12 else 0)
    month = 1 if d.month == 12 else d.month + 1
    return date(year, month, 1)


def month_filename(symbol: str, interval: str, month_start: date) -> str:
    return f"{symbol}-{interval}-{month_start:%Y-%m}.zip"


def day_filename(symbol: str, interval: str, day: date) -> str:
    return f"{symbol}-{interval}-{day:%Y-%m-%d}.zip"


def build_targets(symbol: str, interval: str, start_day: date, end_day: date) -> list[ArchiveTarget]:
    today_utc = datetime.now(timezone.utc).date()
    current_month_start = month_floor(today_utc)
    targets: list[ArchiveTarget] = []

    month_cursor = month_floor(start_day)
    while month_cursor <= end_day:
        month_end_day = date(
            month_cursor.year,
            month_cursor.month,
            calendar.monthrange(month_cursor.year, month_cursor.month)[1],
        )
        if month_end_day < start_day:
            month_cursor = next_month(month_cursor)
            continue

        if month_cursor < current_month_start:
            name = month_filename(symbol, interval, month_cursor)
            targets.append(
                ArchiveTarget(
                    period_type="monthly",
                    key=f"{month_cursor:%Y-%m}",
                    url=f"{BASE_URL}/monthly/klines/{symbol}/{interval}/{name}",
                )
            )
        else:
            day_cursor = max(start_day, month_cursor)
            day_last = min(end_day, today_utc)
            while day_cursor <= day_last:
                name = day_filename(symbol, interval, day_cursor)
                targets.append(
                    ArchiveTarget(
                        period_type="daily",
                        key=f"{day_cursor:%Y-%m-%d}",
                        url=f"{BASE_URL}/daily/klines/{symbol}/{interval}/{name}",
                    )
                )
                day_cursor += timedelta(days=1)

        month_cursor = next_month(month_cursor)

    return targets


def cache_path(cache_dir: str, target: ArchiveTarget) -> str:
    filename = os.path.basename(target.url)
    subdir = os.path.join(cache_dir, target.period_type)
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, filename)


def download_file(target: ArchiveTarget, destination: str, force: bool) -> None:
    if os.path.exists(destination) and not force:
        return

    print(f"Downloading {target.url}")
    try:
        with urllib.request.urlopen(target.url) as response, open(destination, "wb") as fh:
            fh.write(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise FileNotFoundError(f"Archive not found: {target.url}") from exc
        raise


def timestamp_to_datetime(value: int) -> pd.Timestamp:
    unit = "us" if value >= 1_000_000_000_000_000 else "ms"
    return pd.to_datetime(value, unit=unit, utc=True).tz_convert(None)


def read_zip_csv(path: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        members = [name for name in zf.namelist() if name.endswith(".csv")]
        if not members:
            raise ValueError(f"No CSV found inside {path}")
        with zf.open(members[0]) as fh:
            content = fh.read().decode("utf-8")

    reader = csv.reader(io.StringIO(content))
    rows = list(reader)
    if not rows:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    if df.iloc[0]["open_time"] == "open_time":
        df = df.iloc[1:].reset_index(drop=True)
    return df


def normalize_klines(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamps", "open", "high", "low", "close", "volume", "amount"])

    df = df.copy()
    numeric_cols = ["open_time", "open", "high", "low", "close", "volume", "quote_asset_volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open_time", "open", "high", "low", "close"])
    df["timestamps"] = df["open_time"].astype("int64").apply(timestamp_to_datetime)
    out = df.rename(columns={"quote_asset_volume": "amount"})[
        ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    ]
    out = out.sort_values("timestamps").drop_duplicates(subset=["timestamps"]).reset_index(drop=True)
    return out


def kronos_output_path(symbol: str, interval: str, start_day: date, end_day: date) -> str:
    filename = f"{symbol.lower()}_{interval}_{start_day:%Y%m%d}_{end_day:%Y%m%d}.csv"
    return os.path.join("data", filename)


def main() -> None:
    args = parse_args()
    symbol = args.symbol.upper()
    interval = args.interval
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval {interval}. Choose from: {sorted(SUPPORTED_INTERVALS)}")

    start_day = parse_date(args.start_date)
    end_day = parse_date(args.end_date) if args.end_date else datetime.now(timezone.utc).date()
    if start_day > end_day:
        raise ValueError("--start-date must be on or before --end-date")

    output_path = args.output or kronos_output_path(symbol, interval, start_day, end_day)
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.isabs(args.cache_dir):
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.cache_dir)
    else:
        cache_dir = args.cache_dir

    targets = build_targets(symbol, interval, start_day, end_day)
    if not targets:
        raise ValueError("No archive targets were generated for the requested range.")

    frames: list[pd.DataFrame] = []
    missing_targets: list[str] = []
    for target in targets:
        destination = cache_path(cache_dir, target)
        try:
            download_file(target, destination, args.force)
            raw_df = read_zip_csv(destination)
            frames.append(normalize_klines(raw_df))
        except FileNotFoundError:
            missing_targets.append(target.url)

    if not frames:
        raise RuntimeError("No kline data was downloaded successfully.")

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values("timestamps").drop_duplicates(subset=["timestamps"]).reset_index(drop=True)
    result = result[(result["timestamps"].dt.date >= start_day) & (result["timestamps"].dt.date <= end_day)]

    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} rows to {output_path}")
    print(f"Date range: {result['timestamps'].min()} -> {result['timestamps'].max()}")
    if missing_targets:
        print("Missing archives:")
        for url in missing_targets:
            print(f"- {url}")


if __name__ == "__main__":
    main()
