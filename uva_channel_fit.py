"""Week 5 step 2 -- fit the ns-3 channel to the real UVA links.

The ns-3 grid places nodes synthetically, so what it needs from UVA is not the
geometry but how a real link's SNR moves over time.  Fitted here, on the
mid-range links only (Week 2 tiers: SNR neither saturated nor censored, SF7):

  slow        per link, the hourly median SNR minus the link's own median:
              its standard deviation and its autocorrelation at 1-48 h lags,
              summarised by an exponential correlation time tau (AR(1) in hours).
              This is what ns-3 will replay as Gauss-Markov fading per link.
  fast        packet SNR minus its hour's median: within-hour scatter.
  empirical   quantiles of the slow and fast residuals, so ns-3 can inject the
              real distribution instead of a Gaussian.

Day-block bootstrap intervals throughout (hours in a day are not independent).

    python uva_channel_fit.py --build ./build --out ./week5
    python uva_channel_fit.py --selftest
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

LAGS_H = (1, 2, 3, 6, 12, 24, 48)
N_BOOT = 300
TX_ASSUMED_DBM = 20.0
QUANTILES = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99)


def hourly(part: pd.DataFrame) -> pd.Series:
    h = part.set_index("ts")["snr"].resample("1h").median()
    return h - part["snr"].median()


def acf(x: pd.Series, lag: int) -> float:
    a, b = x.to_numpy()[:-lag], x.to_numpy()[lag:]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 50:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def tau_from_acf(rho: dict) -> float:
    """Exponential fit log rho(lag) = c - lag / tau over the lags with rho > 0.05.
    The intercept c absorbs the within-hour noise left in the hourly medians,
    which scales every lag's correlation by the same factor."""
    pts = []
    for lag, r in sorted(rho.items()):          # lags up to the first that has decayed
        if not np.isfinite(r) or r <= 0.05:
            break
        pts.append((lag, r))
    if len(pts) < 2:
        return np.nan
    lags, rs = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts])
    slope = np.polyfit(lags, np.log(rs), 1)[0]
    return float(-1 / slope) if slope < 0 else np.inf


def link_stats(part: pd.DataFrame) -> dict:
    h = hourly(part)
    hour_med = part.set_index("ts")["snr"].groupby(pd.Grouper(freq="1h")).transform("median")
    fast = (part.set_index("ts")["snr"] - hour_med).dropna()
    rho = {lag: acf(h, lag) for lag in LAGS_H}
    return {"sigma_slow": float(np.nanstd(h)), "sigma_fast": float(fast.std()), "rho": rho,
            "tau_h": tau_from_acf(rho), "slow": h.dropna().to_numpy(), "fast": fast.to_numpy(),
            "days": h.dropna().index.normalize().unique()}


def boot_ci(values_by_day: dict, stat, n=N_BOOT, seed=5) -> tuple:
    rng = np.random.default_rng(seed)
    days = list(values_by_day)
    out = []
    for _ in range(n):
        pick = rng.choice(len(days), len(days), replace=True)
        v = np.concatenate([values_by_day[days[i]] for i in pick])
        out.append(stat(v))
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def fit(rec: pd.DataFrame, tiers: pd.DataFrame) -> dict:
    """Temporal statistics on the links whose SNR can move (Week 2 tiers mid-range
    and near-ceiling).  No path-loss law: UVA's two receiving sites cannot
    identify one (a first version fitted 4 links and got an exponent interval
    of 0.75-8.5); the simulator uses the module's standard urban model."""
    res = {"links": [], "tiers": tiers.round(3).to_dict("records")}
    use = tiers[tiers["tier"].isin(["mid_range", "near_ceiling"])]
    slow_all, fast_all, slow_by_day = [], [], {}
    sf7 = rec[rec["sf"] == 7]
    for s, g in use[["sensor", "gateway"]].itertuples(index=False):
        part = sf7[(sf7["sensor"] == s) & (sf7["gateway"] == g)][["ts", "snr"]].dropna()
        if len(part) < 1000:
            res.setdefault("skipped", []).append(f"{s}->{g}: {len(part)} SF7 receptions")
            continue
        part = part.sort_values("ts")
        st = link_stats(part)
        res["links"].append({"sensor": s, "gateway": g, "rx": len(part), "sigma_slow": st["sigma_slow"],
                             "sigma_fast": st["sigma_fast"], "tau_h": st["tau_h"],
                             "rho": {str(k): v for k, v in st["rho"].items()}})
        h = hourly(part).dropna()
        for day, v in h.groupby(h.index.normalize()):
            slow_by_day[(s, g, day)] = v.to_numpy()
        slow_all.append(st["slow"])
        fast_all.append(st["fast"])
    if res["links"]:
        L = pd.DataFrame(res["links"])
        slow, fast = np.concatenate(slow_all), np.concatenate(fast_all)
        res["pooled"] = {
            "sigma_slow_db": float(np.std(slow)),
            "sigma_slow_ci": boot_ci(slow_by_day, np.std),
            "sigma_fast_db": float(np.std(fast)),
            "tau_h_median": float(np.nanmedian(L["tau_h"].replace(np.inf, np.nan))),
            "rho_median": {str(k): float(np.nanmedian([l["rho"][str(k)] for l in res["links"]])) for k in LAGS_H},
            "slow_quantiles": {str(q): float(np.quantile(slow, q)) for q in QUANTILES},
            "fast_quantiles": {str(q): float(np.quantile(fast, q)) for q in QUANTILES},
        }
    return res


def report(res: dict) -> str:
    out = ["# Week 5 -- UVA channel fit for ns-3 (temporal statistics)", "",
           "Link tiers (Week 2 rules); only mid_range and near_ceiling links are used:", "",
           "| link | rx | mean SNR | p05 SNR | share at ceiling | PDR | tier |", "| --- " * 7 + "|"]
    for t in res["tiers"]:
        out.append(f"| {t['sensor']}->{t['gateway']} | {t['rx']} | {t['snr_mean']} | {t['snr_p05']} | "
                   f"{t['at_ceiling']} | {t['pdr']} | {t['tier']} |")
    for sk in res.get("skipped", []):
        out.append(f"\nskipped {sk}")
    out += ["", "| link | rx | sigma slow (hourly) dB | sigma fast (within hour) dB | tau h | " +
            " | ".join(f"rho {l} h" for l in LAGS_H) + " |", "| --- " * (5 + len(LAGS_H)) + "|"]
    for l in res["links"]:
        out.append(f"| {l['sensor']}->{l['gateway']} | {l['rx']} | {l['sigma_slow']:.2f} | {l['sigma_fast']:.2f} | "
                   f"{l['tau_h']:.1f} | " + " | ".join(f"{l['rho'][str(k)]:.2f}" for k in LAGS_H) + " |")
    if "pooled" in res:
        q = res["pooled"]
        out += ["", f"Pooled: slow sigma {q['sigma_slow_db']:.2f} dB [{q['sigma_slow_ci'][0]:.2f}, "
                f"{q['sigma_slow_ci'][1]:.2f}], fast sigma {q['sigma_fast_db']:.2f} dB, "
                f"median tau {q['tau_h_median']:.1f} h.",
                "Median autocorrelation: " + ", ".join(f"{k} h {v:.2f}" for k, v in q["rho_median"].items())]
    else:
        out += ["", "NO LINK QUALIFIED -- see the tier table above."]
    return "\n".join(out) + "\n"


def selftest() -> None:
    # Fixtures, not data; nothing here becomes a result.
    rng = np.random.default_rng(1)
    n_h, tau, sig = 24 * 120, 6.0, 3.0
    a = math.exp(-1 / tau)
    x = np.zeros(n_h)
    for i in range(1, n_h):
        x[i] = a * x[i - 1] + sig * math.sqrt(1 - a * a) * rng.standard_normal()
    ts = pd.date_range("2024-01-01", periods=n_h * 12, freq="5min", tz="UTC")
    snr = np.repeat(x, 12) + 1.5 * rng.standard_normal(n_h * 12) + 5
    st = link_stats(pd.DataFrame({"ts": ts, "snr": snr}))
    assert abs(st["tau_h"] - tau) < 1.5, st["tau_h"]
    assert abs(st["sigma_slow"] - sig) < 0.5, st["sigma_slow"]
    assert abs(st["sigma_fast"] - 1.5) < 0.2, st["sigma_fast"]
    print(f"selftest ok (planted tau 6 h, sigma 3 / 1.5 dB -> {st['tau_h']:.1f} h, "
          f"{st['sigma_slow']:.2f} / {st['sigma_fast']:.2f} dB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="./build")
    ap.add_argument("--out", default="./week5")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    import uva_week2 as w2
    rec = pd.read_parquet(Path(a.build) / "uva_receptions.parquet")
    pdr = pd.read_parquet(Path(a.build) / "uva_pdr.parquet")
    tiers = w2.link_tiers(rec, pdr)
    res = fit(rec, tiers)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "CHANNEL_FIT.md").write_text(report(res))
    (out / "channel_fit.json").write_text(json.dumps(res, indent=1, default=str))
    print(report(res))


if __name__ == "__main__":
    main()
