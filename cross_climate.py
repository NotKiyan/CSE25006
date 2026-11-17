"""The Week 2 estimator, run on every dataset in the standard format.

Same method as uva_week2.py -- each series compared only with itself, week
fixed effects (season, foliage, steps), hour-of-day fixed effects (the
daily rhythm of people and interference), week-block bootstrap -- applied
per unit (node, gateway, SF), then pooled per dataset and gateway with one
shared bootstrap.

Results are in dB per 10 °C so datasets with different climates sit on one
scale.  Three metrics per unit say WHERE an effect lives:
  SNR and RSSI fall, noise floor flat   -> the radio path (or the node's TX)
  RSSI and noise fall together, SNR flat -> the gateway's gain / RSSI scale
  SNR flat, RSSI flat                   -> no effect

    python cross_climate.py --standard ./standard --out ./cross
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import uva_config as cfg
from datasets_common import SITES
from uva_week2 import FixedEffects, ci, md_table, pooled

PER = 10.0                      # report everything per 10 °C
MIN_RX = 1000


def hourly(part: pd.DataFrame, tz: str) -> pd.DataFrame:
    if "co2_ppm" not in part:                 # standard files written before CO2 existed
        part = part.assign(co2_ppm=np.nan)
    h = (part.assign(hour=part["ts"].dt.floor("1h"))
             .groupby("hour").agg(snr=("snr", "mean"), rssi=("rssi", "mean"),
                                  noise=("noise", "mean"), temp_c=("temp_c", "mean"),
                                  rh_pct=("rh_pct", "mean"), co2_ppm=("co2_ppm", "mean"),
                                  n=("snr", "size")))
    local = h.index.tz_convert(tz)
    iso = local.isocalendar()
    h["hour_local"] = local.hour.to_numpy()
    h["week"] = iso["year"].astype("int64").to_numpy() * 100 + iso["week"].astype("int64").to_numpy()
    return h.dropna(subset=["temp_c"])


# Thermal noise in 125 kHz is about -123 dBm, plus a 6-10 dB receiver noise
# figure.  An implied noise floor above -100 dBm is not noise at all: the
# gateway's SNR reading has topped out on a strong link, so RSSI - SNR just
# tracks RSSI.  On such a unit a real signal loss would ALSO look like
# "RSSI and noise fall, SNR flat", so its SNR and noise are not evidence.
COMPRESSED_NOISE_DBM = -100.0


def tier(part: pd.DataFrame) -> tuple[str, float, float]:
    """Same rules as Week 2, with the floor of THIS unit's SF, plus compression."""
    snr = part["snr"].dropna()
    if part["noise"].median() > COMPRESSED_NOISE_DBM:
        return "compressed", np.nan, float(snr.quantile(0.05))
    sf = int(part["sf"].iloc[0])
    # Unknown SF (greenhouse): assume the lowest floor, SF12, so no unit is
    # called censored on a guess.  The report carries that caveat.
    floor = cfg.SF_FLOOR_DB.get(sf, -20.0)
    ceiling = float(snr.quantile(0.999))
    sat = float((snr >= ceiling - cfg.SAT_WINDOW_DB).mean())
    p05 = float(snr.quantile(0.05))
    if sat > cfg.SAT_SHARE_SATURATED:
        t = "saturated"
    elif p05 < floor + cfg.MARGINAL_HEADROOM_DB:
        t = "marginal"
    elif sat > cfg.SAT_SHARE_NEAR:
        t = "near_ceiling"
    else:
        t = "mid_range"
    return t, sat, p05


def analyse_dataset(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    site = SITES[df["dataset"].iloc[0]]
    has_rh = df["rh_pct"].notna().mean() > 0.5
    xs = ["temp_c", "rh_pct"] if has_rh else ["temp_c"]
    # Occupancy control where the dataset has it (Siegen): people raise CO2,
    # warm the room and attenuate signals.
    if "co2_ppm" in df and df["co2_ppm"].notna().mean() > 0.5:
        xs.append("co2_ppm")
    units, fits = [], {}
    for unit, part in df.groupby("unit", sort=True):
        if len(part) < MIN_RX:
            continue
        t, sat, p05 = tier(part)
        h = hourly(part, site.tz)
        row = {"dataset": site.name, "unit": unit, "node": part["node"].iloc[0],
               "gateway": part["gateway"].iloc[0], "sf": int(part["sf"].iloc[0]),
               "tier": t, "rx": len(part), "snr_mean": float(part["snr"].mean()),
               "snr_p05": p05, "at_ceiling": sat,
               "noise_median_dbm": float(part["noise"].median()),
               "temp_p05": float(part["temp_c"].quantile(.05)),
               "temp_p95": float(part["temp_c"].quantile(.95))}
        for metric in ("snr", "rssi", "noise"):
            fe = FixedEffects(h, metric, xs)
            if not fe.ok:
                continue
            fits[(unit, metric)] = fe
            lo, hi = ci(fe.boot(rng)[:, 0])
            row[f"{metric}_per10"] = float(fe.beta()[0]) * PER
            row[f"{metric}_lo"] = lo * PER
            row[f"{metric}_hi"] = hi * PER
            row["weeks"] = fe.n_weeks
        units.append(row)
    u = pd.DataFrame(units)

    # Pooled, per gateway: same metric rules as Week 2.
    snr_ok = {"mid_range", "near_ceiling"}
    noise_ok = {"mid_range", "near_ceiling", "marginal"}
    pools = []
    for gw, g in u.groupby("gateway"):
        row = {"dataset": site.name, "gateway": gw, "units": len(g),
               "temp_source": site.temp_source,
               "temp_p05": float(g["temp_p05"].median()), "temp_p95": float(g["temp_p95"].median())}
        for metric, allowed in (("snr", snr_ok), ("rssi", None), ("noise", noise_ok)):
            keep = g if allowed is None else g[g["tier"].isin(allowed)]
            r = pooled([fits.get((un, metric)) for un in keep["unit"]], rng, PER)
            row[f"{metric}_units"] = r["links"]
            row[f"{metric}_per10"], row[f"{metric}_lo"], row[f"{metric}_hi"] = r["swing"], r["lo"], r["hi"]
        row["compressed_units"] = int((g["tier"] == "compressed").sum())
        row["reading"] = reading(row)
        pools.append(row)
    return u, pd.DataFrame(pools)


def reading(r: dict) -> str:
    """Name where the effect sits, only when the intervals support it."""
    def clear(m):          # interval excludes zero
        lo, hi = r.get(f"{m}_lo", np.nan), r.get(f"{m}_hi", np.nan)
        return np.isfinite(lo) and (hi < 0 or lo > 0)

    def small(m, bound=0.5):  # interval sits inside +-bound dB per 10 °C
        lo, hi = r.get(f"{m}_lo", np.nan), r.get(f"{m}_hi", np.nan)
        return np.isfinite(lo) and -bound < lo and hi < bound

    if clear("rssi") and clear("noise") and small("snr"):
        return "gateway: RSSI and noise floor move together, SNR does not"
    if clear("snr") and clear("noise"):
        # Received signal power = noise floor + SNR, so the two parts add.
        return ("both: SNR falls (path or node) and the gateway's scale moves; "
                "signal power change = noise change + SNR change")
    if clear("snr") and clear("rssi") and small("noise"):
        return "path or transmitter: SNR and RSSI move, noise floor does not"
    if small("snr") and small("rssi"):
        return "no effect within ±0.5 dB per 10 °C"
    if not np.isfinite(r.get("snr_lo", np.nan)):
        return "SNR not measurable here (units saturated, censored or compressed)"
    return "mixed or unresolved"


def report(units: pd.DataFrame, pools: pd.DataFrame) -> str:
    def band(df, m):
        return [f"{a:+.2f} [{b:+.2f}, {c:+.2f}]" if np.isfinite(a) else "—"
                for a, b, c in zip(df[f"{m}_per10"], df[f"{m}_lo"], df[f"{m}_hi"])]
    p = pools.copy()
    p["temp range °C"] = [f"{a:.0f} to {b:.0f}" for a, b in zip(p["temp_p05"], p["temp_p95"])]
    for m in ("snr", "rssi", "noise"):
        p[f"{m.upper()} dB/10°C"] = band(p, m)
    L = ["# Climate effect across datasets", "",
         "dB per 10 °C, pooled per dataset and gateway, 95% week-block intervals. "
         "Same estimator everywhere: week and hour-of-day fixed effects, each series "
         "compared only with itself. Temperature range is p05 to p95.", "",
         md_table(p, ["dataset", "gateway", "temp_source", "temp range °C", "units",
                      "SNR dB/10°C", "RSSI dB/10°C", "NOISE dB/10°C", "reading"],
                  ["dataset", "gateway", "temp from", "temp range °C", "units", "SNR",
                   "RSSI", "noise floor", "reading"]), "",
         "SNR and noise floor are pooled only over units where SNR measures the channel: "
         "units pinned at the SNR ceiling, censored at the floor, or COMPRESSED (implied noise "
         f"floor above {COMPRESSED_NOISE_DBM:.0f} dBm, so SNR has topped out) are left out of "
         "those two columns. RSSI uses every unit.", "",
         "## Dataset notes", ""] + [f"- **{n}**: {SITES[n].note}" for n in sorted(units["dataset"].unique())] + [
         "", "## Units", ""]
    u = units.copy()
    for m in ("snr", "rssi", "noise"):
        if f"{m}_per10" not in u:
            u[f"{m}_per10"] = u[f"{m}_lo"] = u[f"{m}_hi"] = np.nan
        u[m.upper()] = band(u, m)
    L.append(md_table(u.sort_values(["dataset", "gateway", "unit"]),
                      ["dataset", "unit", "tier", "rx", "snr_mean", "noise_median_dbm",
                       "SNR", "RSSI", "NOISE"],
                      ["dataset", "unit", "tier", "rx", "SNR mean", "noise dBm",
                       "SNR /10°C", "RSSI /10°C", "noise /10°C"],
                      fmt={"snr_mean": "{:.1f}", "noise_median_dbm": "{:.1f}"}))
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--standard", default="./standard")
    ap.add_argument("--out", default="./cross")
    ap.add_argument("--only", nargs="*", help="dataset names to run (default: all found)")
    args = ap.parse_args()
    rng = np.random.default_rng(cfg.BOOT_SEED)
    files = sorted(Path(args.standard).glob("*.parquet"))
    if args.only:
        files = [f for f in files if f.stem in args.only]
    all_u, all_p = [], []
    for f in files:
        print(f"{f.stem} ...")
        u, p = analyse_dataset(pd.read_parquet(f), rng)
        all_u.append(u)
        all_p.append(p)
        for r in p.itertuples():
            print(f"  {r.gateway}: SNR {r.snr_per10:+.2f}  RSSI {r.rssi_per10:+.2f}  "
                  f"noise {r.noise_per10:+.2f} dB/10°C -> {r.reading}")
    units, pools = pd.concat(all_u, ignore_index=True), pd.concat(all_p, ignore_index=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    units.to_csv(out / "units.csv", index=False)
    pools.to_csv(out / "pooled.csv", index=False)
    (out / "CROSS_CLIMATE.md").write_text(report(units, pools))
    print(f"wrote {out}/CROSS_CLIMATE.md")


if __name__ == "__main__":
    main()
