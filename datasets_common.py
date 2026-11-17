"""One table format for every dataset, so one analysis runs on all of them.

Standard reception table, one row per received packet at one gateway:

  dataset   name of the dataset
  unit      the series the analysis treats as one link: node, gateway and SF
            (SF is part of the unit because each SF has its own decoding
            floor, and Medellin rotates SF by design)
  node, gateway, sf
  ts        UTC
  snr, rssi           dB, dBm
  noise               implied noise floor, dBm:  rssi - 10*log10(1 + 10**(snr/10))
  temp_c, rh_pct      the temperature the dataset pairs with this packet
  temp_source         "node" (sensor on the device) or "station" (one site station)

Each dataset also declares where it is (time zone, latitude, longitude) so
local hour and sun angle are computed correctly for it.

    python datasets_common.py medellin "<path to the zip>" --out ./standard
    python datasets_common.py uva ./build --out ./standard
"""

from __future__ import annotations

import argparse
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STANDARD = ["dataset", "unit", "node", "gateway", "sf", "ts", "snr", "rssi",
            "noise", "temp_c", "rh_pct", "co2_ppm", "temp_source"]


@dataclass(frozen=True)
class Site:
    name: str
    tz: str
    lat: float
    lon: float
    temp_source: str
    note: str


SITES = {
    "uva": Site("uva", "America/New_York", 38.033, -78.511, "station",
                "UVA campus; one weather station; US915"),
    "medellin": Site("medellin", "America/Bogota", 6.25, -75.57, "node",
                     "Medellin, Colombia; DHT22 on each node; US915; "
                     "authors' 'No Outliers V2' release"),
    "siegen": Site("siegen", "Europe/Berlin", 50.87, 8.02, "room",
                   "Indoor office, University of Siegen; room sensors with CO2 as an "
                   "occupancy control; EU868; narrow indoor temperature range -- a "
                   "control, not a heat test"),
    "greenhouse": Site("greenhouse", "Europe/Brussels", 51.40, 4.77, "node",
                       "Antwerp tomato greenhouses (BE, NL); imec OCTA-Connect on-board "
                       "sensor; SF not logged (ADR likely on)"),
}


def implied_noise(rssi: pd.Series, snr: pd.Series) -> pd.Series:
    """Noise floor from packet RSSI (signal + noise) and SNR, valid at any SNR."""
    return rssi - 10 * np.log10(1 + 10 ** (snr / 10))


def finish(df: pd.DataFrame, site: Site) -> pd.DataFrame:
    df = df.copy()
    df["dataset"] = site.name
    df["temp_source"] = site.temp_source
    df["noise"] = implied_noise(df["rssi"], df["snr"])
    if "co2_ppm" not in df:
        df["co2_ppm"] = np.nan
    # Physically impossible sensor readings, for every dataset (Siegen has a
    # room at 174.9 °C).  Blanked, not dropped: the radio reading is fine.
    for col, lo, hi in (("temp_c", -30, 70), ("rh_pct", 0, 100), ("co2_ppm", 250, 10000)):
        bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        if bad.any():
            print(f"  {int(bad.sum())} impossible {col} readings set to missing")
            df.loc[bad, col] = np.nan
    sf_label = df["sf"].astype(int).astype(str).where(df["sf"].astype(int) > 0, "?")
    df["unit"] = df["node"].astype(str) + "→" + df["gateway"].astype(str) + "@SF" + sf_label
    df = df.dropna(subset=["ts", "snr", "rssi"])
    return df[STANDARD].sort_values(["unit", "ts"]).reset_index(drop=True)


# ==========================================================================
# Time zone: decided from the data, never assumed
# ==========================================================================
def localise(naive: pd.Series, temp: pd.Series, tz: str,
             clues: dict[str, pd.Series] | None = None,
             shift_hours: float = 0.0) -> tuple[pd.Series, str]:
    """Return UTC timestamps, deciding whether naive times are local or UTC.

    Air temperature peaks in the early afternoon everywhere, so the reading
    that puts the daily peak between 11:00 and 17:00 LOCAL time is the
    right one.  If neither does, stop rather than guess.
    """
    if shift_hours:
        naive = naive + pd.Timedelta(hours=shift_hours)
        print(f"  clock shifted by {shift_hours:+g} h, as instructed")
    as_local = naive.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    as_utc = naive.dt.tz_localize("UTC")
    peaks = {}
    for label, ts in (("local", as_local), ("utc", as_utc)):
        hour = ts.dt.tz_convert(tz).dt.hour
        peaks[label] = int(temp.groupby(hour).mean().idxmax())
    ok = {k: v for k, v in peaks.items() if 11 <= v <= 17}
    if len(ok) == 2:
        # Both readings are plausible: the site is only an hour or two from
        # UTC.  Take the one nearer the usual 14:30 peak.  An error of that
        # size cannot bias the fixed-effects estimate: hour of day enters
        # only as a label, and relabelling every hour by the same amount
        # changes nothing.
        dist = {k: abs(v - 14.5) for k, v in ok.items()}
        if dist["local"] == dist["utc"]:
            raise SystemExit("time zone tie: both readings put the peak equally close to 14:30")
        best = min(dist, key=dist.get)
        ok = {best: ok[best]}
        print(f"  note: both readings are plausible here (peaks {peaks['local']}:00 vs "
              f"{peaks['utc']}:00); chose the one nearer 14:30. Harmless for this estimator.")
    if len(ok) != 1:
        print(f"  ! cannot tell the time zone: temperature peaks at stored hour "
              f"{int(temp.groupby(naive.dt.hour).mean().idxmax())}. Clues, by STORED hour:")
        for name, series in (clues or {}).items():
            prof = series.groupby(naive.dt.hour).mean()
            print(f"    {name}: highest at {int(prof.idxmax())}:00, lowest at {int(prof.idxmin())}:00")
        raise SystemExit("send these lines back; rerun with --shift-hours N once the offset is known")
    choice = next(iter(ok))
    print(f"  time zone: stored times are {choice.upper()} "
          f"(temperature peaks at {ok[choice]}:00 local under that reading; "
          f"{peaks['utc' if choice == 'local' else 'local']}:00 under the other)")
    return (as_local if choice == "local" else as_utc), choice


# ==========================================================================
# Loaders
# ==========================================================================
def clock_drift_check(naive: pd.Series, temp: pd.Series, node: pd.Series) -> bool:
    """Is each node's clock offset CONSTANT?  Peak hour per node per month.

    A constant offset per node only relabels that node's hours, and every
    unit gets its own hour-of-day effects, so it cannot bias the estimate.
    A drifting clock moves the peak from month to month, and that would.
    """
    months = naive.dt.to_period("M")
    tab = (pd.DataFrame({"node": node, "month": months, "hour": naive.dt.hour, "t": temp})
             .groupby(["node", "month", "hour"])["t"].median()
             .groupby(level=[0, 1]).idxmax().map(lambda k: k[2]).unstack("month"))
    print("  peak-temperature hour (stored clock), per node and month:")
    print("   " + tab.to_string().replace("\n", "\n   "))
    spread = tab.apply(lambda r: min((r.max() - r.min()), 24 - (r.max() - r.min())), axis=1)
    stable = bool((spread <= 3).all())
    print(f"  peak hour moves by at most {int(spread.max())} h within any node -> "
          f"{'constant offsets: safe for this estimator' if stable else 'DRIFTING clocks: not safe'}")
    return stable


def load_medellin(path: Path, shift_hours: float = 0.0, clock: str = "auto") -> pd.DataFrame:
    """Gonzalez-Palacio et al., MDPI Data 8(1):4 (2023).

    Dates are day-first ("4/11/2021 16:24" is 4 November 2021: the campaign
    ran October 2021 to March 2022, so month-first would fall outside it).
    The file has no frame counter, so PDR cannot be rebuilt; it is SNR, RSSI
    and noise only.  `pn` in the file equals the implied noise floor; it is
    recomputed here from RSSI and SNR and checked against the file's value.
    """
    site = SITES["medellin"]
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(m for m in z.namelist() if m.lower().endswith(".csv"))
            raw = pd.read_csv(io.BytesIO(z.read(name)))
    else:
        raw = pd.read_csv(path)
    print(f"  {len(raw):,} rows")

    naive = pd.to_datetime(raw["timestamp"], dayfirst=True, errors="coerce")
    bad = int(naive.isna().sum())
    if bad:
        print(f"  {bad} unparseable timestamps dropped")
    span = (naive.min(), naive.max())
    if not (pd.Timestamp("2021-09-01") <= span[0] and span[1] <= pd.Timestamp("2022-04-30")):
        raise SystemExit(f"dates {span} fall outside the Oct 2021 - Mar 2022 campaign; "
                         "the day-first reading is wrong for this file")
    if clock == "as-is":
        # Temperature travels in the same packet as RSSI and SNR, so the
        # pairing is right whatever the clock says.  Keep stored times as
        # labels; refuse if the per-node offsets drift.
        if not clock_drift_check(naive, raw["temperature"], raw["device_id"].astype(str)):
            raise SystemExit("clocks drift: Medellin cannot be used with hour-of-day controls")
        ts = naive.dt.tz_localize("UTC")
        print("  clock: stored times kept as labels (not true UTC); fine for cross_climate.py only")
    clues = {f"temperature {n}": raw["temperature"].where(raw["device_id"] == n)
             for n in sorted(raw["device_id"].astype(str).unique())}
    clues["humidity (all)"] = raw["rh"].where(raw["rh"] >= 5)
    if "pm2_5" in raw:
        clues["PM2.5, morning traffic (all)"] = raw["pm2_5"]
    if "pn" in raw:
        clues["noise floor pn (all)"] = raw["pn"]
    if clock != "as-is":
        ts, _ = localise(naive, raw["temperature"], site.tz, clues, shift_hours)

    df = pd.DataFrame({
        "node": raw["device_id"].astype(str), "gateway": "GW",
        "sf": pd.to_numeric(raw["sf"], errors="coerce"),
        "ts": ts, "snr": pd.to_numeric(raw["snr"], errors="coerce"),
        "rssi": pd.to_numeric(raw["rssi"], errors="coerce"),
        "temp_c": pd.to_numeric(raw["temperature"], errors="coerce"),
        "rh_pct": pd.to_numeric(raw["rh"], errors="coerce"),
    })
    # A relative humidity of 1% in a tropical valley is a sensor glitch.
    glitch = df["rh_pct"] < 5
    if glitch.any():
        print(f"  {int(glitch.sum())} humidity readings under 5% set to missing")
        df.loc[glitch, "rh_pct"] = np.nan

    out = finish(df, site)
    if "pn" in raw.columns:
        diff = (implied_noise(df["rssi"], df["snr"]) - raw["pn"]).abs().median()
        print(f"  implied noise vs the file's pn column: median difference {diff:.3f} dB")
    return out


def load_greenhouse(folder: Path, shift_hours: float = 0.0) -> pd.DataFrame:
    """Singh et al., Zenodo 10.5281/zenodo.5793685 (Sensors 22(4):1326, 2022).

    One sheet per sensor (AF16..AF42 in Greenhouse-1, AF49..AF67 in
    Greenhouse-2), columns Date, Time, Onboard Temperature, Onboard
    Humidity, RSSI, SNR.  Greenhouse-2 sheets carry extra pivot-table
    columns, and there are summary sheets (TEMP, RSSI, ...): both ignored.

    Trimmed to deployment: Greenhouse-1 sensors sat in a room until
    2020-04-17 06:38 (dataset README); Greenhouse-2 uses its START-END sheet.
    SF is not logged.  SNR down to -15.5 dB means SF10 or above was in use,
    so ADR was probably moving SF: these series are scored on RSSI and the
    noise floor first, and SNR is reported with that caveat.
    """
    site = SITES["greenhouse"]
    folder = Path(folder)
    frames = []
    for path in sorted(folder.glob("Greenhouse-*.xlsx")):
        if "Transformed" in path.name:
            continue                                   # hourly means, not packets
        gh = "GH1" if "Greenhouse-1" in path.name else "GH2"
        sheets = pd.read_excel(path, sheet_name=None)
        windows = {}
        if "START-END" in sheets:
            se = sheets["START-END"]
            windows = {str(n): (pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(days=1))
                       for n, a, b in zip(se["Sensor"], se["Start Date"], se["End Date"])}
        for name, sh in sheets.items():
            if not str(name).upper().startswith("AF") or "Onboard Temperature" not in sh:
                continue
            sh = sh[["Date", "Time", "Onboard Temperature", "Onboard Humidity", "RSSI", "SNR"]].dropna(
                subset=["Date", "Time"])
            naive = pd.to_datetime(sh["Date"].astype(str).str[:10] + " " + sh["Time"].astype(str),
                                   errors="coerce")
            part = pd.DataFrame({"naive": naive, "node": str(name), "gateway": gh,
                                 "temp_c": pd.to_numeric(sh["Onboard Temperature"], errors="coerce"),
                                 "rh_pct": pd.to_numeric(sh["Onboard Humidity"], errors="coerce"),
                                 "rssi": pd.to_numeric(sh["RSSI"], errors="coerce"),
                                 "snr": pd.to_numeric(sh["SNR"], errors="coerce")})
            if gh == "GH1":
                part = part[part["naive"] >= pd.Timestamp("2020-04-17 06:38")]
            elif str(name) in windows:
                a, b = windows[str(name)]
                part = part[(part["naive"] >= a) & (part["naive"] < b)]
            frames.append(part)
        print(f"  {path.name}: {sum(1 for n in sheets if str(n).upper().startswith('AF'))} sensor sheets")
    raw = pd.concat(frames, ignore_index=True).dropna(subset=["naive"])
    bad_t = (raw["temp_c"] < -10) | (raw["temp_c"] > 60)
    bad_h = (raw["rh_pct"] < 1) | (raw["rh_pct"] > 100)
    print(f"  {len(raw):,} rows; {int(bad_t.sum())} impossible temperatures and "
          f"{int(bad_h.sum())} impossible humidities set to missing")
    raw.loc[bad_t, "temp_c"] = np.nan
    raw.loc[bad_h, "rh_pct"] = np.nan

    ts, _ = localise(raw["naive"], raw["temp_c"], site.tz, None, shift_hours)
    raw["ts"] = ts
    raw["sf"] = -1                                      # not logged
    return finish(raw, site)


def load_siegen(path: Path, shift_hours: float = 0.0) -> pd.DataFrame:
    """Obiri & Van Laerhoven, Zenodo 10.5281/zenodo.15349731 (IEEE Access 2025).

    File 2.aggregated_measurements_data.csv.  Times carry an explicit +00:00,
    so they are UTC and nothing is guessed.  CO2 is kept: occupancy raises
    CO2, warms the room and attenuates signals, so it is the control that
    stops people being credited to temperature.  The file's n_power column is
    compared with the implied noise floor, as for Medellin.
    """
    site = SITES["siegen"]
    use = ["time", "device_id", "co2", "humidity", "temperature", "rssi", "snr", "SF", "n_power"]
    raw = pd.read_csv(path, usecols=lambda c: c in use)
    print(f"  {len(raw):,} rows, {raw['device_id'].nunique()} devices")
    ts = pd.to_datetime(raw["time"], utc=True, format="mixed", errors="coerce")
    if shift_hours:
        ts = ts + pd.Timedelta(hours=shift_hours)
    if ts.isna().any():
        print(f"  {int(ts.isna().sum())} unparseable timestamps dropped")
    df = pd.DataFrame({
        "node": raw["device_id"].astype(str), "gateway": "GW",
        "sf": pd.to_numeric(raw["SF"], errors="coerce"), "ts": ts,
        "snr": pd.to_numeric(raw["snr"], errors="coerce"),
        "rssi": pd.to_numeric(raw["rssi"], errors="coerce"),
        "temp_c": pd.to_numeric(raw["temperature"], errors="coerce"),
        "rh_pct": pd.to_numeric(raw["humidity"], errors="coerce"),
        "co2_ppm": pd.to_numeric(raw["co2"], errors="coerce"),
    }).dropna(subset=["sf"])
    hour = df["ts"].dt.tz_convert(site.tz).dt.hour
    print(f"  room temperature peaks at {int(df['temp_c'].groupby(hour).mean().idxmax())}:00 local, "
          f"CO2 at {int(df['co2_ppm'].groupby(hour).mean().idxmax())}:00 (information only)")
    out = finish(df, site)
    if "n_power" in raw:
        diff = (implied_noise(df["rssi"], df["snr"]) - raw.loc[df.index, "n_power"]).abs().median()
        print(f"  implied noise vs the file's n_power column: median difference {diff:.3f} dB")
    return out


def load_uva(build: Path) -> pd.DataFrame:
    """UVA, from the Week 1 build: station temperature, SF7 as recorded."""
    site = SITES["uva"]
    cols = ["ts", "sensor", "gateway", "sf", "snr", "rssi", "temp_c", "rh_pct"]
    r = pd.read_parquet(Path(build) / "uva_receptions.parquet", columns=cols)
    r = r.rename(columns={"sensor": "node"})
    r = r[r["node"] != "sensor10"]                       # 17-20 days only
    r["node"] = r["node"].str.replace("sensor", "s", regex=False)
    r["gateway"] = r["gateway"].str.replace("gateway", "", regex=False)
    return finish(r, site)


LOADERS = {"medellin": load_medellin, "greenhouse": load_greenhouse, "siegen": load_siegen,
           "uva": load_uva}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(LOADERS))
    ap.add_argument("path")
    ap.add_argument("--out", default="./standard")
    ap.add_argument("--shift-hours", type=float, default=0.0,
                    help="move stored times by N hours before localising (clock fixes)")
    ap.add_argument("--clock", choices=["auto", "as-is"], default="auto",
                    help="medellin only: 'as-is' keeps stored times as labels after a drift check")
    args = ap.parse_args()
    print(f"loading {args.dataset} ...")
    loader = LOADERS[args.dataset]
    if args.dataset == "uva":
        df = loader(Path(args.path))
    elif args.dataset == "medellin":
        df = loader(Path(args.path), shift_hours=args.shift_hours, clock=args.clock)
    else:
        df = loader(Path(args.path), shift_hours=args.shift_hours)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{args.dataset}.parquet"
    df.to_parquet(dest, index=False)
    t = df["temp_c"].dropna()
    print(f"  {len(df):,} receptions, {df['unit'].nunique()} units, "
          f"{df['ts'].min():%Y-%m-%d} to {df['ts'].max():%Y-%m-%d}")
    print(f"  temperature ({SITES[args.dataset].temp_source}): p05 {t.quantile(.05):.1f}, "
          f"median {t.median():.1f}, p95 {t.quantile(.95):.1f}, max {t.max():.1f} °C")
    print(f"  wrote {dest}")


if __name__ == "__main__":
    main()
