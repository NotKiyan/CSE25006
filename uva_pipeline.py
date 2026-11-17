"""UVA LoRaWAN dataset -> clean, leakage-safe tables for the climate-aware ADR work.

Produces four tables plus build_stats.json (uva_pdr.parquet: one row per
sensor-gateway-hour with reconstructed PDR and that hour's SNR stats):

  uva_receptions.parquet  one row per (uplink, gateway) reception.
                          Everything the raw files had, plus a stable
                          uplink_id, epoch, local time, solar elevation,
                          link distance and placement flags.

  uva_packets.parquet     one row per UNIQUE uplink (deduplicated across
                          gateways).  Carries n_hear, best-gateway SNR, the
                          SNR spread across gateways, and which gateways
                          heard it.  This is the multi-gateway table.

  uva_weather.parquet     hourly weather on a regular index, with the
                          outage gaps marked rather than silently absent.

Run order:
    python uva_pipeline.py --inspect   # look before you leap: schemas only
    python uva_pipeline.py --build     # write the three tables

Both take --root /path/to/unzipped/dataset  (the folder that CONTAINS
`dataset/`), or set UVA_ROOT in the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import uva_config as cfg


# ==========================================================================
# Locating the data
# ==========================================================================
def find_dataset(root: Path) -> dict[str, Path]:
    """Locate the pieces of the archive, wherever the user unzipped it."""
    root = Path(root).expanduser().resolve()
    if not root.exists():
        sys.exit(f"--root does not exist: {root}")

    def first(patterns: list[str]) -> Path | None:
        for pat in patterns:
            hits = sorted(root.glob(pat))
            if hits:
                return hits[0]
        return None

    combined = first([
        "dataset/lorawan_metadata/lorawan_combined_dataset.parquet",
        "**/lorawan_combined_dataset.parquet",
    ])
    weather = first([
        "dataset/weather/deployment_weather.parquet",
        "**/deployment_weather.parquet",
    ])
    summary = first([
        "dataset/summary_statistics/lorawan_summary_by_pair.csv",
        "**/lorawan_summary_by_pair.csv",
    ])
    pairs = sorted(root.glob("**/sensor*_gateway*.parquet"))

    if combined is None and not pairs:
        sys.exit(
            f"No LoRaWAN parquet found under {root}.\n"
            "Point --root at the folder that contains `dataset/`."
        )
    return {"root": root, "combined": combined, "weather": weather,
            "summary": summary, "pairs": pairs}


# ==========================================================================
# Loading and canonicalising
# ==========================================================================
def canonicalise(df: pd.DataFrame) -> pd.DataFrame:
    """Map README column names onto the short names used downstream."""
    rename: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in cfg.COLUMN_MAP:
            rename[col] = cfg.COLUMN_MAP[key]
            continue
        for kw, short in cfg.COLUMN_KEYWORDS:          # fallback
            if kw in key:
                rename[col] = short
                break
    df = df.rename(columns=rename)
    df = df.loc[:, ~df.columns.duplicated()]
    missing = [c for c in cfg.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(
            f"Missing required columns after canonicalisation: {missing}\n"
            f"Columns seen: {list(df.columns)}\n"
            "Add the real names to COLUMN_MAP in uva_config.py."
        )
    return df


def load_receptions(paths: dict) -> pd.DataFrame:
    """Load every reception event.  Prefers the combined file."""
    if paths["combined"] is not None:
        df = pd.read_parquet(paths["combined"])
        source = paths["combined"].name
    else:
        frames = []
        for p in paths["pairs"]:
            part = pd.read_parquet(p)
            # The per-pair files may omit the alias columns; recover them
            # from the filename, which the README fixes as sensorXX_gatewayY.
            s_alias, _, g_alias = p.stem.partition("_")
            lowered = {str(c).strip().lower() for c in part.columns}
            if "sensor alias" not in lowered:
                part["Sensor Alias"] = s_alias
            if "gateway alias" not in lowered:
                part["Gateway Alias"] = g_alias
            frames.append(part)
        df = pd.concat(frames, ignore_index=True)
        source = f"{len(paths['pairs'])} per-pair files"

    df = canonicalise(df)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    for c in ("rssi", "snr", "airtime"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("sf", "counter", "bw", "freq", "n_hear_reported"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    df.attrs["source"] = source
    return df


# ==========================================================================
# Cleaning
# ==========================================================================
def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop rows that cannot be used.  Every drop is counted, not silent."""
    n0 = len(df)
    log: dict[str, int] = {"rows_in": n0}

    def drop(mask: pd.Series, reason: str) -> None:
        nonlocal df
        n = int(mask.sum())
        if n:
            log[f"dropped_{reason}"] = n
            df = df.loc[~mask]

    drop(df["ts"].isna(), "bad_timestamp")
    drop(df["counter"].isna(), "null_counter")
    drop(df["sensor"].isna() | df["gateway"].isna(), "null_alias")
    drop(df["snr"].isna() & df["rssi"].isna(), "no_signal_metric")

    # Physically implausible values.  LoRa SNR lives roughly in [-25, +15] dB
    # and RSSI in [-150, 0] dBm; anything outside is a logging artefact.
    drop(df["snr"].notna() & ((df["snr"] < -30) | (df["snr"] > 20)), "snr_out_of_range")
    drop(df["rssi"].notna() & ((df["rssi"] < -150) | (df["rssi"] > 0)), "rssi_out_of_range")

    # Aliases the README does not define.
    known_s, known_g = set(cfg.SENSORS), set(cfg.GATEWAYS)
    drop(~df["sensor"].isin(known_s), "unknown_sensor")
    drop(~df["gateway"].isin(known_g), "unknown_gateway")

    # Exact duplicate rows: same link, same counter, same instant.
    dup = df.duplicated(subset=["sensor", "gateway", "counter", "ts"], keep="first")
    drop(dup, "exact_duplicate_rows")

    df = df.sort_values(["sensor", "ts", "gateway"]).reset_index(drop=True)
    log["rows_out"] = len(df)
    return df, log


# ==========================================================================
# Uplink identity: dedup across gateways, then frame-counter epochs
# ==========================================================================
def assign_uplinks(df: pd.DataFrame) -> pd.DataFrame:
    """Group receptions into unique uplinks, then number counter epochs.

    An uplink is every reception of one (sensor, counter) that falls within
    UPLINK_GROUP_TOLERANCE_S of the previous reception of that same
    (sensor, counter).  Receptions of OTHER counters arriving in between do
    not break the group.  The first version required receptions to sit next
    to each other in the time-sorted stream, which split one transmission in
    two whenever a gateway logged its copy late; that is what drove the
    65% n_hear agreement on the first real run.  A counter reused after a
    reboot is hours away and lands in its own group.

    If a device uses confirmed uplinks, its retransmissions reuse the same
    counter within seconds and are folded into one uplink here.  That is
    the right unit for PDR, but n_hear then means "gateways that heard any
    attempt".

    An epoch is a stretch over which the counter never goes backwards.  A
    drop means the device rebooted or the counter wrapped; the true count
    across that boundary is unknown, so PDR is only computed inside one
    epoch.  Epochs are detected on the uplink sequence (one row per uplink,
    at its first reception), so a late duplicate can never look like a
    reset.
    """
    # Step 1 -- cluster receptions into transmissions.  Within one
    # (sensor, counter), receptions closer than the tolerance are one
    # transmission; a later reuse of the same counter (after a reboot) is
    # far away in time and becomes its own cluster.  Crucially, rows of
    # OTHER counters landing in between no longer break a cluster -- that
    # adjacency rule was the original bug.
    df = df.sort_values(["sensor", "counter", "ts"]).reset_index(drop=True)
    same_key = (df["sensor"] == df["sensor"].shift()) & (df["counter"] == df["counter"].shift())
    gap_s = df["ts"].diff().dt.total_seconds()
    new_tx = ~same_key | (gap_s > cfg.UPLINK_GROUP_TOLERANCE_S)
    df["uplink_id"] = new_tx.cumsum().astype("int64")

    # Step 2 -- epochs, on the transmission sequence in time order (one row
    # per uplink, at its first reception), so duplicates can never look
    # like a counter going backwards.
    first = (df.groupby("uplink_id", sort=False)
               .agg(sensor=("sensor", "first"), counter=("counter", "first"), ts=("ts", "min"))
               .reset_index().sort_values(["sensor", "ts", "counter"]))
    sensor_boundary = first["sensor"] != first["sensor"].shift()
    went_back = ((first["counter"].diff() < 0) & ~sensor_boundary).fillna(False)
    first["epoch"] = went_back.groupby(first["sensor"]).cumsum().astype("int64")

    df = df.merge(first[["uplink_id", "epoch"]], on="uplink_id", how="left")
    return df.sort_values(["sensor", "ts", "gateway"]).reset_index(drop=True)


def build_packets(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse receptions to one row per unique uplink.

    This is the table that makes multi-gateway diversity visible: n_hear,
    the best gateway's SNR, and how far apart the gateways' views are.
    """
    spec = {
        "sensor": ("sensor", "first"),
        "ts": ("ts", "min"),
        "counter": ("counter", "first"),
        "epoch": ("epoch", "first"),
        "sf": ("sf", "first"),
        "n_hear": ("gateway", "nunique"),
        "snr_best": ("snr", "max"),
        "snr_worst": ("snr", "min"),
        "snr_mean": ("snr", "mean"),
        "rssi_best": ("rssi", "max"),
        "gateways": ("gateway", lambda s: "|".join(sorted(set(s)))),
    }
    if "n_hear_reported" in df.columns:
        spec["n_hear_reported"] = ("n_hear_reported", "max")
    if "airtime" in df.columns:
        spec["airtime"] = ("airtime", "first")

    pk = df.groupby("uplink_id", sort=False).agg(**spec).reset_index()
    pk["n_hear"] = pk["n_hear"].astype("int64")
    pk["snr_spread"] = pk["snr_best"] - pk["snr_worst"]

    # Which gateway heard it best.  Uplinks with no SNR anywhere keep NaN.
    with_snr = df.dropna(subset=["snr"])
    if len(with_snr):
        best = with_snr.loc[with_snr.groupby("uplink_id", sort=False)["snr"].idxmax(),
                            ["uplink_id", "gateway"]]
        pk = pk.merge(best.rename(columns={"gateway": "gateway_best"}),
                      on="uplink_id", how="left")
    else:
        pk["gateway_best"] = pd.NA

    if "n_hear_reported" in pk.columns:
        pk["n_hear_agrees"] = pk["n_hear"] == pk["n_hear_reported"]
    return pk


# ==========================================================================
# PDR reconstruction
# ==========================================================================
def reconstruct_pdr(packets: pd.DataFrame, receptions: pd.DataFrame,
                    bin_size: str = cfg.PDR_BIN) -> pd.DataFrame:
    """Per sensor-gateway-hour delivery ratio, from frame-counter gaps.

    Expected uplinks in a bin run from just after the previous hour's last
    counter to this hour's last counter, so losses on a bin edge are
    counted.  Only the first bin after a gap (or an epoch start) falls back
    to this bin's own first counter, and only there can losses hide.  A bin
    containing an epoch boundary is split into one row per epoch, never
    scored across the reset.
    """
    pk = packets.copy()
    pk["bin"] = pk["ts"].dt.floor(bin_size)

    span = (pk.groupby(["sensor", "epoch", "bin"], sort=False)
              .agg(cmin=("counter", "min"), cmax=("counter", "max"),
                   seen_any=("uplink_id", "nunique"))
              .reset_index().sort_values(["sensor", "epoch", "bin"]))

    # Chain each bin to the one before it.  Counting from this bin's own
    # first counter would miss every packet lost between the previous bin's
    # last counter and this bin's first -- a loss that falls on a bin edge
    # would vanish and PDR would read high.  When the previous hour exists
    # in the same epoch, the expected count starts right after its last
    # counter instead.
    grp = span.groupby(["sensor", "epoch"], sort=False)
    prev_cmax = grp["cmax"].shift()
    prev_bin = grp["bin"].shift()
    contiguous = (span["bin"] - prev_bin) == pd.Timedelta(bin_size)
    span["cstart"] = np.where(contiguous, prev_cmax + 1, span["cmin"])
    span["expected"] = (span["cmax"] - span["cstart"] + 1).astype("int64")

    # Guard rails.  A bin with too few expected uplinks is noise; a bin with
    # an implausibly long gap is a device outage, not a delivery failure.
    span = span[span["expected"] >= cfg.PDR_MIN_EXPECTED]
    span = span[span["expected"] <= cfg.PDR_MAX_GAP]

    rec = receptions.copy()
    # Bin each reception by its UPLINK's first-seen time, not its own: a
    # copy logged late at one gateway must count in the same bin as the
    # transmission it belongs to.
    rec = rec.merge(pk[["uplink_id", "bin"]], on="uplink_id", how="left")
    per_gw = rec.groupby(["sensor", "epoch", "bin", "gateway"], sort=False).agg(
        received=("uplink_id", "nunique"),
        snr_mean=("snr", "mean"),
        snr_p10=("snr", lambda s: s.quantile(0.10)),
        rssi_mean=("rssi", "mean"),
        sf=("sf", "max"),           # one SF per sensor-hour in practice
    ).reset_index()

    # Every gateway is scored in every bin the sensor was alive, so a gateway
    # that heard nothing gets PDR 0 rather than a missing row.
    grid = span[["sensor", "epoch", "bin", "expected", "seen_any"]].merge(
        pd.DataFrame({"gateway": sorted(cfg.GATEWAYS)}), how="cross")
    out = grid.merge(per_gw, on=["sensor", "epoch", "bin", "gateway"], how="left")
    out["received"] = out["received"].fillna(0).astype("int64")

    out["pdr"] = (out["received"] / out["expected"]).clip(upper=1.0)
    out["pdr_any"] = (out["seen_any"] / out["expected"]).clip(upper=1.0)
    return out


# ==========================================================================
# Time, sun, geometry
# ==========================================================================
def solar_elevation(lat_deg: float, lon_deg: float, ts_utc: pd.Series) -> np.ndarray:
    """Solar elevation in degrees (NOAA low-precision, ~0.1 deg).

    This is the control CASM lacks.  Its feature vector carries Month but no
    hour-of-day, so temperature there partly stands in for time of day.  With
    solar elevation in the model, a temperature coefficient has to earn its
    place against the diurnal cycle instead of impersonating it.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(ts_utc, utc=True))
    n = idx.to_julian_date().to_numpy() - 2451545.0

    mean_long = np.radians((280.460 + 0.9856474 * n) % 360.0)
    mean_anom = np.radians((357.528 + 0.9856003 * n) % 360.0)
    ecl_long = mean_long + np.radians(1.915) * np.sin(mean_anom) \
        + np.radians(0.020) * np.sin(2 * mean_anom)
    obliquity = np.radians(23.439 - 4e-7 * n)

    decl = np.arcsin(np.sin(obliquity) * np.sin(ecl_long))
    right_asc = np.arctan2(np.cos(obliquity) * np.sin(ecl_long), np.cos(ecl_long))

    gmst = (18.697374558 + 24.06570982441908 * n) % 24.0
    local_sidereal = (gmst + lon_deg / 15.0) % 24.0
    hour_angle = np.radians((local_sidereal - np.degrees(right_asc) / 15.0) * 15.0)

    lat = np.radians(lat_deg)
    sin_elev = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    return np.degrees(np.arcsin(np.clip(sin_elev, -1.0, 1.0)))


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add local time, solar elevation, link geometry and placement flags."""
    local = df["ts"].dt.tz_convert(cfg.LOCAL_TZ)
    df["ts_local"] = local
    df["hour_local"] = local.dt.hour.astype("int16")
    df["month"] = local.dt.month.astype("int8")

    # One campus, so one solar elevation series is enough.
    centre_lat = float(np.mean([s["lat"] for s in cfg.SENSORS.values()]))
    centre_lon = float(np.mean([s["lon"] for s in cfg.SENSORS.values()]))
    df["solar_elev"] = solar_elevation(centre_lat, centre_lon, df["ts"])
    df["is_daylight"] = df["solar_elev"] > 0

    dist = cfg.link_distances()
    df["distance_m"] = [
        dist.get((s, g), np.nan) for s, g in zip(df["sensor"], df["gateway"])
    ]
    df["sensor_indoor"] = df["sensor"].map(lambda s: cfg.SENSORS[s]["indoor"])
    df["gateway_indoor"] = df["gateway"].map(lambda g: cfg.GATEWAYS[g]["indoor"])
    return df


# ==========================================================================
# Weather
# ==========================================================================
def _is_fahrenheit(name: str, values: pd.Series) -> bool:
    low = name.lower().replace(" ", "")
    if "(f)" in low or low.endswith("_f") or "degf" in low:
        return True
    if "(c)" in low or low.endswith("_c"):
        return False
    v = values.dropna()
    return bool(len(v)) and v.median() > 50       # no climate has a 50 °C median


def canonicalise_weather(w: pd.DataFrame) -> tuple[pd.DataFrame, list[str], set[str]]:
    """Rename every weather column to a short name that carries its unit.

    Each column is converted to SI once, here: every temperature-like column
    (air, dew point, heat index, feels-like, wind chill) to °C, pressure to
    hPa, wind to m/s, and the station's ACCUMULATED rain counter to rain in
    mm per reading, so it can be summed per hour.  Columns no rule matches
    are kept under a sanitised name.  Wind direction is dropped: a plain
    mean of angles is wrong (the mean of 350° and 10° is not 180°) and
    nothing downstream uses it.

    Returns the frame, the numeric value columns, and the subset to SUM
    (not average) when resampling to hours.
    """
    rules = [   # (keywords that must all appear, canonical name, kind)
        (("dew",), "dew_point_c", "temp"),
        (("heat", "index"), "heat_index_c", "temp"),
        (("feels",), "feels_like_c", "temp"),
        (("chill",), "wind_chill_c", "temp"),
        (("temp",), "temp_c", "temp"),
        (("humid",), "rh_pct", "plain"),
        (("pressure",), "pressure_hpa", "pressure"),
        (("rain",), "rain_mm", "rain"),
        (("wind", "average"), "wind_avg_ms", "wind"),
        (("wind", "avg"), "wind_avg_ms", "wind"),
        (("wind", "speed"), "wind_speed_ms", "wind"),
        (("wind", "direction"), None, "drop"),
    ]
    out = pd.DataFrame({"ts": w["ts"].to_numpy()})
    sum_cols: set[str] = set()
    for col in w.columns:
        if col == "ts" or not pd.api.types.is_numeric_dtype(w[col]):
            continue
        low = str(col).lower()
        match = next((r for r in rules if all(k in low for k in r[0])), None)
        series = w[col].astype(float).to_numpy()
        if match is None:
            name = "".join(ch if ch.isalnum() else "_" for ch in low).strip("_")
            out[name] = series
            continue
        _, name, kind = match
        if kind == "drop" or name in out.columns:
            continue
        if kind == "temp":
            if _is_fahrenheit(str(col), w[col]):
                series = (series - 32.0) * 5.0 / 9.0
                print(f"  weather: '{col}' -> {name} (°F converted to °C)")
        elif kind == "pressure":
            if np.nanmedian(series) < 40:                    # inHg, not hPa
                series = series * 33.8639
                print(f"  weather: '{col}' -> {name} (inHg converted to hPa)")
        elif kind == "wind":
            if "mph" in low:
                series = series * 0.44704
        elif kind == "rain":
            inches = "(in" in low.replace(" ", "") or "inch" in low
            if "accum" in low or "total" in low:
                # Counter -> increments.  A drop means the counter reset,
                # and the new reading is the rain since that reset.
                inc = np.diff(series, prepend=np.nan)
                reset = inc < 0
                inc[reset] = series[reset]
                inc[0] = 0.0
                series = inc
                print(f"  weather: '{col}' -> {name} (accumulated counter "
                      "converted to per-reading increments)")
            if inches:
                series = series * 25.4
            sum_cols.add(name)
        out[name] = series
    value_cols = [c for c in out.columns if c != "ts"]
    return out, value_cols, sum_cols


def load_weather(path: Path | None) -> tuple[pd.DataFrame | None, list[dict]]:
    """Hourly weather on a regular index, with outage gaps named."""
    if path is None:
        return None, []
    w = pd.read_parquet(path)

    # Find the time column without assuming its name.  A real datetime
    # column wins; a name match is the fallback, and it matches whole words
    # so a column like "update_count" can't be mistaken for a date.
    tcol = next((c for c in w.columns if pd.api.types.is_datetime64_any_dtype(w[c])), None)
    if tcol is None:
        tcol = next((c for c in w.columns
                     if any(k in str(c).lower().replace("-", "_").split("_")
                            for k in ("time", "timestamp", "date", "datetime", "ts"))),
                    None)
    if tcol is None:
        print("  ! weather: no time column found; columns =", list(w.columns))
        return w, []

    w[tcol] = pd.to_datetime(w[tcol], utc=True, errors="coerce")
    w = w.dropna(subset=[tcol]).rename(columns={tcol: "ts"}).sort_values("ts")

    w, value_cols, sum_cols = canonicalise_weather(w)
    agg = {c: ("sum" if c in sum_cols else "mean") for c in value_cols}
    hourly = w.set_index("ts")[value_cols].resample("1h").agg(agg)
    # A summed column reads 0 in an hour with no readings at all; that hour
    # is missing, not dry.
    counts = w.set_index("ts")[value_cols[0]].resample("1h").count()
    for c in sum_cols:
        hourly.loc[counts == 0, c] = np.nan
    full = hourly.reindex(pd.date_range(hourly.index.min(), hourly.index.max(),
                                        freq="1h", tz="UTC"))

    missing = full[value_cols].isna().all(axis=1)
    gaps, run_start = [], None
    for stamp, is_missing in missing.items():
        if is_missing and run_start is None:
            run_start = stamp
        elif not is_missing and run_start is not None:
            hours = (stamp - run_start).total_seconds() / 3600
            if hours >= cfg.WEATHER_GAP_MIN_HOURS:
                gaps.append({"start": str(run_start), "end": str(stamp),
                             "hours": round(hours, 1)})
            run_start = None
    if run_start is not None:
        hours = (full.index[-1] - run_start).total_seconds() / 3600
        if hours >= cfg.WEATHER_GAP_MIN_HOURS:
            gaps.append({"start": str(run_start), "end": str(full.index[-1]),
                         "hours": round(hours, 1)})

    full = full.reset_index(names="ts")
    full["weather_missing"] = missing.to_numpy()
    return full, gaps


def weather_column(weather: pd.DataFrame, *keywords: str) -> str | None:
    """Find a weather column by keyword without assuming the README's name."""
    for c in weather.columns:
        low = str(c).lower()
        if any(k in low for k in keywords):
            return c
    return None


def join_weather(rec: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    """Attach the hourly weather row covering each reception.

    The join is on the floor of the UTC hour, so a reception gets the mean
    weather of the hour it happened in.  That mean includes readings from
    later in the same hour.  That is fine for measuring how climate moves
    a link (Week 2), but it is mild look-ahead for forecasting.  Week 3
    features must use the PREVIOUS hour's weather.  Receptions inside a
    weather outage keep NaN weather and are flagged, not dropped -- Week 2
    decides what to exclude, this step only records the fact.
    """
    if weather is None:
        rec["weather_missing"] = True
        return rec
    w = weather.copy()
    w["__hour"] = pd.to_datetime(w["ts"], utc=True).dt.floor("1h")
    w = w.drop(columns=["ts"]).drop_duplicates(subset="__hour")
    rec = rec.copy()
    rec["__hour"] = rec["ts"].dt.floor("1h")
    out = rec.merge(w, on="__hour", how="left").drop(columns="__hour")
    out["weather_missing"] = out["weather_missing"].fillna(True).astype(bool)
    return out


# ==========================================================================
# Modes
# ==========================================================================
def inspect(paths: dict) -> None:
    """Print real schemas before anything is computed from them."""
    print(f"root     : {paths['root']}")
    print(f"combined : {paths['combined']}")
    print(f"weather  : {paths['weather']}")
    print(f"per-pair : {len(paths['pairs'])} files")
    print()

    target = paths["combined"] or (paths["pairs"][0] if paths["pairs"] else None)
    if target:
        df = pd.read_parquet(target)
        print(f"--- {Path(target).name}: {len(df):,} rows ---")
        print(df.dtypes.to_string())
        print()
        print(df.head(3).to_string())
        print()
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                print(f"  {c}: min={df[c].min()} max={df[c].max()} nulls={df[c].isna().sum()}")
        print()

    if paths["weather"]:
        w = pd.read_parquet(paths["weather"])
        print(f"--- {paths['weather'].name}: {len(w):,} rows ---")
        print(w.dtypes.to_string())
        print()
        print(w.head(3).to_string())


def build(paths: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    stats: dict = {"root": str(paths["root"]), "pipeline_version": cfg.PIPELINE_VERSION}

    print("loading receptions ...")
    raw = load_receptions(paths)
    stats["source"] = raw.attrs.get("source")
    print(f"  {len(raw):,} rows from {stats['source']}")

    print("cleaning ...")
    rec, drop_log = clean(raw)
    stats["cleaning"] = drop_log
    for k, v in drop_log.items():
        print(f"  {k}: {v:,}")

    print("assigning uplinks and epochs ...")
    rec = assign_uplinks(rec)
    rec = enrich(rec)
    stats["n_receptions"] = int(len(rec))
    stats["n_uplinks"] = int(rec["uplink_id"].nunique())
    stats["n_epochs"] = int(rec.groupby(["sensor", "epoch"]).ngroups)
    stats["epochs_per_sensor"] = {s: int(e) + 1 for s, e in
                                  rec.groupby("sensor")["epoch"].max().items()}
    print(f"  {stats['n_uplinks']:,} unique uplinks from {stats['n_receptions']:,} receptions")

    print("building the packet table ...")
    packets = build_packets(rec)
    if "n_hear_agrees" in packets.columns:
        agree = float(packets["n_hear_agrees"].mean())
        stats["n_hear_agreement"] = round(agree, 4)
        print(f"  reconstructed n_hear agrees with the README column on "
              f"{agree:.1%} of uplinks")
        # Which way do they disagree?  Reported > reconstructed on most
        # mismatches means the network had gateways beyond the three
        # published; reconstructed > reported means over-merging here.
        xt = pd.crosstab(packets["n_hear"], packets["n_hear_reported"],
                         rownames=["reconstructed"], colnames=["reported"])
        stats["n_hear_crosstab"] = {f"{r}->{c}": int(xt.loc[r, c])
                                    for r in xt.index for c in xt.columns if xt.loc[r, c]}
        print(xt.to_string())

    print("reconstructing PDR ...")
    pdr = reconstruct_pdr(packets, rec)
    stats["n_pdr_bins"] = int(len(pdr))

    print("loading weather ...")
    weather, gaps = load_weather(paths["weather"])
    stats["weather_gaps"] = gaps
    for g in gaps:
        print(f"  gap: {g['start'][:10]} -> {g['end'][:10]}  ({g['hours']:.0f} h)")

    print("joining weather to receptions ...")
    rec = join_weather(rec, weather)
    covered = float((~rec["weather_missing"]).mean())
    stats["receptions_with_weather"] = round(covered, 4)
    print(f"  {covered:.1%} of receptions fall in an hour with weather")

    print("writing ...")
    rec.to_parquet(outdir / "uva_receptions.parquet", index=False)
    packets.to_parquet(outdir / "uva_packets.parquet", index=False)
    pdr.to_parquet(outdir / "uva_pdr.parquet", index=False)
    if weather is not None:
        weather.to_parquet(outdir / "uva_weather.parquet", index=False)
    (outdir / "build_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(f"done -> {outdir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("UVA_ROOT", "."),
                    help="folder containing `dataset/` (or set UVA_ROOT)")
    ap.add_argument("--out", default="./build", help="where to write the tables")
    ap.add_argument("--inspect", action="store_true", help="print schemas and exit")
    ap.add_argument("--build", action="store_true", help="write the clean tables")
    args = ap.parse_args()

    paths = find_dataset(Path(args.root))
    if args.inspect or not args.build:
        inspect(paths)
        if not args.build:
            return
    build(paths, Path(args.out))


if __name__ == "__main__":
    main()
