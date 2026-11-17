"""Inspect every dataset file before a loader is written for it.

Walks a folder (CSV, XLSX, and CSVs inside ZIPs, without extracting), and
for each table prints what a loader has to know: columns and types, three
sample rows, row count, the range of every numeric column, and the time
span and sampling interval of every time column.  Temperature-like columns
get a percentile line, because heat coverage decides what a dataset is for.

It also runs the UVA time-zone check: if the UVA weather is really in UTC,
the warmest hour of the day lands around 19-20 UTC (3-4 pm in Virginia).

    python inspect_datasets.py --root ../datasets --uva-build ./build > datasets_report.txt

Nothing is modified.  Send back datasets_report.txt.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

TIME_WORDS = ("time", "date", "timestamp", "ts")
TEMP_WORDS = ("temp",)
MAX_PRINT_COLS = 40
CHUNK = 500_000


def is_time_col(name: str) -> bool:
    low = str(name).lower()
    return any(w in low for w in TIME_WORDS) and "airtime" not in low and "toa" not in low


def describe(df: pd.DataFrame, label: str, total_rows: int | None = None,
             head: pd.DataFrame | None = None) -> None:
    print(f"\n=== {label}")
    print(f"rows: {total_rows if total_rows is not None else len(df):,} | columns: {df.shape[1]}")
    cols = list(df.columns)
    print("columns:", cols[:MAX_PRINT_COLS], "..." if len(cols) > MAX_PRINT_COLS else "")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print((head if head is not None else df).head(3).to_string(max_colwidth=28))
    print("-- per column (min | p01 | median | p99 | max | null%):")
    for c in cols[:MAX_PRINT_COLS]:
        s = df[c]
        null = f"{s.isna().mean():.1%}"
        if is_time_col(c) and not pd.api.types.is_numeric_dtype(s):
            t = pd.to_datetime(s, errors="coerce", utc=False)
            if t.notna().mean() > 0.5:
                step = t.dropna().sort_values().diff().dt.total_seconds()
                tz = getattr(t.dt, "tz", None)
                print(f"  {c} [time, tz={tz}]: {t.min()} -> {t.max()} | median step "
                      f"{step.median():.0f}s | unparseable {t.isna().mean():.1%} | "
                      f"sample raw value {s.dropna().astype(str).iloc[0]!r}")
                continue
        if pd.api.types.is_numeric_dtype(s):
            v = s.dropna()
            if len(v):
                q = v.quantile([0.01, 0.5, 0.99])
                print(f"  {c}: {v.min():.4g} | {q.iloc[0]:.4g} | {q.iloc[1]:.4g} | "
                      f"{q.iloc[2]:.4g} | {v.max():.4g} | {null}")
                if any(w in str(c).lower() for w in TEMP_WORDS):
                    p = v.quantile([0.05, 0.25, 0.75, 0.95])
                    hot30, hot35 = (v >= 30).mean(), (v >= 35).mean()
                    print(f"     temp: p05 {p.iloc[0]:.1f}, p25 {p.iloc[1]:.1f}, p75 {p.iloc[2]:.1f}, "
                          f"p95 {p.iloc[3]:.1f} | share >=30: {hot30:.2%}, >=35: {hot35:.2%}")
            else:
                print(f"  {c}: all null")
        else:
            vc = s.astype(str).value_counts()
            print(f"  {c} [text]: {s.nunique():,} distinct, top { {k: int(v) for k, v in vc.head(5).items()} } | {null}")


def read_csv_full(handle_factory, label: str) -> None:
    """Read a CSV in chunks so a 2 GB file never sits in memory at once."""
    first, rows, frames = None, 0, []
    for chunk in pd.read_csv(handle_factory(), chunksize=CHUNK, low_memory=False):
        rows += len(chunk)
        if first is None:
            first = chunk
        # Keep a strided sample across the whole file for the ranges.
        frames.append(chunk.iloc[:: max(1, len(chunk) // 20_000)])
    if first is None:
        print(f"\n=== {label}: empty")
        return
    sample = pd.concat(frames, ignore_index=True)
    describe(sample, label, total_rows=rows, head=first)
    print("  (ranges from an evenly spread sample of the whole file)")


def inspect_path(p: Path) -> None:
    size = p.stat().st_size / 1e6
    suf = p.suffix.lower()
    if suf == ".csv":
        read_csv_full(lambda: open(p, "rb"), f"{p} ({size:.1f} MB)")
    elif suf in (".xlsx", ".xls"):
        sheets = pd.read_excel(p, sheet_name=None)
        for name, df in sheets.items():
            describe(df, f"{p} [sheet {name!r}] ({size:.1f} MB)")
    elif suf == ".zip":
        with zipfile.ZipFile(p) as z:
            members = z.namelist()
            print(f"\n=== {p} ({size:.1f} MB) contains {len(members)} files:")
            for m in members[:50]:
                print("   ", m)
            for m in members:
                ml = m.lower()
                if ml.endswith(".csv"):
                    read_csv_full(lambda m=m: io.BytesIO(z.read(m)), f"{p} :: {m}")
                elif ml.endswith((".xlsx", ".xls")):
                    for name, df in pd.read_excel(io.BytesIO(z.read(m)), sheet_name=None).items():
                        describe(df, f"{p} :: {m} [sheet {name!r}]")


def uva_timezone_check(build: Path) -> None:
    f = build / "uva_weather.parquet"
    print("\n=== UVA weather time-zone check")
    if not f.exists():
        print(f"  {f} not found; skipped")
        return
    w = pd.read_parquet(f)
    hour = pd.to_datetime(w["ts"], utc=True).dt.hour
    prof = w.groupby(hour)["temp_c"].mean()
    print("  mean temp_c by UTC hour:", " ".join(f"{h}:{t:.1f}" for h, t in prof.items()))
    peak = int(prof.idxmax())
    verdict = ("UTC confirmed (peak at 3-4 pm Virginia time)" if 17 <= peak <= 22 else
               "looks like LOCAL time stored as UTC -- the weather join is shifted"
               if 12 <= peak <= 16 else "unclear -- send this line back")
    print(f"  warmest UTC hour: {peak} -> {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../datasets")
    ap.add_argument("--uva-build", default="./build")
    ap.add_argument("--skip", nargs="*", default=["unsorted_combined"],
                    help="filename fragments to skip (the 2 GB raw dump by default)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    files = sorted(f for f in root.rglob("*")
                   if f.suffix.lower() in (".csv", ".xlsx", ".xls", ".zip")
                   and not any(s in f.name for s in args.skip))
    print(f"root: {root}\nfound {len(files)} data files:")
    for f in files:
        print(f"  {f.relative_to(root)}  ({f.stat().st_size / 1e6:.1f} MB)")
    for f in files:
        try:
            inspect_path(f)
        except Exception as e:                       # one bad file must not stop the rest
            print(f"\n=== {f}: could not read ({type(e).__name__}: {e})")
    uva_timezone_check(Path(args.uva_build))


if __name__ == "__main__":
    main()
