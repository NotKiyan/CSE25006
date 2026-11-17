"""Week 2 -- the climate gate.

One question: how many dB does a hot afternoon cost a UVA link, once
everything that is NOT temperature has been taken out?  The answer decides
the paper's headline (see the gate in the plan).

What is taken out, and how:

  * Geometry and device offsets (up to ~5 dB between identical sensors):
    every link is compared only with itself.
  * Season, foliage, slow drift, and any step change: week fixed effects.
    A coefficient is learned only from variation INSIDE one week.
  * Time of day (campus activity, interference, people in rooms): hour-of-
    day fixed effects.  So temperature is credited only when a given hour
    is hotter or cooler than that hour usually is that week -- the control
    CASM's model did not have.
  * ADR: SF7 only.  Nine of ten sensors never left it.

Two estimators, which should agree:
  A. Diurnal contrast -- per week, day-minus-night of the metric against
     day-minus-night of temperature, slope across weeks.  The plan's
     primary analysis: foliage is constant inside a week.
  B. Fixed effects -- hourly metric ~ temperature + humidity + hour FE +
     week FE.  The more conservative of the two; the gate uses it.

Each link is scored on the metric that can actually move on it (Week 1 found
saturated and censored links), and every estimate carries a week-block
bootstrap 95% interval, because hours inside a week are not independent.

    python uva_week2.py --build ./build --out ./week2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import uva_config as cfg
import uva_pipeline as up

# Saturated links are scored on PDR, not RSSI: the first UVA run showed RSSI
# at the rooftop gateways moving ~7 dB with temperature while SNR on the same
# links did not move, so RSSI cannot be assumed to track the signal.
TIER_METRIC = {"mid_range": "snr", "near_ceiling": "snr", "saturated": "pdr",
               "marginal": "pdr", "floor": "pdr", "excluded": None}


# ==========================================================================
# Loading
# ==========================================================================
def load(build: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stats = json.loads((build / "build_stats.json").read_text())
    if stats.get("pipeline_version") != cfg.PIPELINE_VERSION:
        raise SystemExit(
            f"{build} was built by pipeline version {stats.get('pipeline_version')}, "
            f"this code expects {cfg.PIPELINE_VERSION} (canonical weather columns). "
            "Rebuild with uva_pipeline.py --build.")

    cols = ["ts", "sensor", "gateway", "snr", "rssi", "sf", "freq", "uplink_id"]
    rec = pd.read_parquet(build / "uva_receptions.parquet", columns=cols)
    pdr = pd.read_parquet(build / "uva_pdr.parquet")
    wx = pd.read_parquet(build / "uva_weather.parquet")
    for need in ("temp_c", "rh_pct"):
        if need not in wx.columns:
            raise SystemExit(f"weather table has no '{need}' column: {list(wx.columns)}")
    return rec, pdr, wx


def weather_hourly(wx: pd.DataFrame) -> pd.DataFrame:
    """Hourly weather indexed by UTC hour, with lags, outages as NaN."""
    w = wx.copy()
    w["hour"] = pd.to_datetime(w["ts"], utc=True)
    w = w.set_index("hour").sort_index()
    keep = [c for c in ("temp_c", "rh_pct", "pressure_hpa", "rain_mm", "dew_point_c")
            if c in w.columns]
    w = w[keep + ["weather_missing"]]
    w.loc[w["weather_missing"], keep] = np.nan
    # The index is regular (the pipeline reindexed it), so shift(k) is k hours.
    w["temp_lag1"] = w["temp_c"].shift(1)
    w["temp_lag3"] = w["temp_c"].shift(3)
    return w.drop(columns="weather_missing")


def time_features(hours: pd.DatetimeIndex) -> pd.DataFrame:
    """Local hour, ISO week, weekday and sun angle for a set of UTC hours."""
    local = hours.tz_convert(cfg.LOCAL_TZ)
    iso = local.isocalendar()
    centre_lat = float(np.mean([s["lat"] for s in cfg.SENSORS.values()]))
    centre_lon = float(np.mean([s["lon"] for s in cfg.SENSORS.values()]))
    mid = pd.Series(hours + pd.Timedelta(minutes=30))
    return pd.DataFrame({
        "hour_local": local.hour.to_numpy(),
        "month": local.month.to_numpy(),
        "week": (iso["year"].astype("int64").to_numpy() * 100
                 + iso["week"].astype("int64").to_numpy()),
        "weekday": (local.dayofweek < 5),
        "solar_elev": up.solar_elevation(centre_lat, centre_lon, mid),
    }, index=hours)


# ==========================================================================
# Link tiers
# ==========================================================================
def link_tiers(rec: pd.DataFrame, pdr: pd.DataFrame) -> pd.DataFrame:
    """Decide, from the data, which metric can move on each link."""
    floor = cfg.SF_FLOOR_DB[cfg.ANALYSIS_SF]
    mean_pdr = pdr.groupby(["sensor", "gateway"])["pdr"].mean()
    rows = []
    for (s, g), part in rec.groupby(["sensor", "gateway"], sort=True):
        snr = part.loc[part["sf"] == cfg.ANALYSIS_SF, "snr"].dropna()
        p = float(mean_pdr.get((s, g), np.nan))
        if len(snr):
            ceiling = float(snr.quantile(0.999))
            sat = float((snr >= ceiling - cfg.SAT_WINDOW_DB).mean())
            p05 = float(snr.quantile(0.05))
        else:
            ceiling = sat = p05 = np.nan

        if s in cfg.EXCLUDE_SENSORS or len(part) < cfg.MIN_RX_PER_LINK:
            tier = "excluded"
        elif p < cfg.FLOOR_PDR:
            tier = "floor"
        elif sat > cfg.SAT_SHARE_SATURATED:
            tier = "saturated"
        elif p05 < floor + cfg.MARGINAL_HEADROOM_DB:
            tier = "marginal"
        elif sat > cfg.SAT_SHARE_NEAR:
            tier = "near_ceiling"
        else:
            tier = "mid_range"
        rows.append({"sensor": s, "gateway": g, "rx": len(part),
                     "snr_mean": float(snr.mean()) if len(snr) else np.nan,
                     "snr_p05": p05, "snr_ceiling": ceiling, "at_ceiling": sat,
                     "pdr": p, "tier": tier, "metric": TIER_METRIC[tier]})
    return pd.DataFrame(rows)


# ==========================================================================
# Per-link hourly table
# ==========================================================================
def link_hours(part: pd.DataFrame, wxh: pd.DataFrame, ceiling: float = np.inf
               ) -> pd.DataFrame:
    """Hourly mean SNR, RSSI and implied noise floor for one link.

    Packet RSSI is total received power, signal plus noise, so
        noise = RSSI - 10*log10(1 + 10**(SNR/10))
    holds at any SNR.  This is the receiver-side test: real propagation loss
    lowers the signal and leaves the noise alone, while a change in the
    gateway's gain or RSSI calibration moves signal and noise together and
    leaves SNR unchanged.  The thermal floor for 125 kHz is -123 dBm before
    any noise figure, so an implied noise floor below that, or one that
    falls in the heat, is a measurement artefact, not physics.  Packets at
    the SNR ceiling are left out of the noise estimate, since a clipped SNR
    gives a wrong noise.
    """
    part = part[part["sf"] == cfg.ANALYSIS_SF].copy()
    ok = part["snr"] < ceiling - cfg.SAT_WINDOW_DB
    part["noise"] = np.where(ok, part["rssi"] - 10 * np.log10(1 + 10 ** (part["snr"] / 10)),
                             np.nan)
    h = (part.assign(hour=part["ts"].dt.floor("1h"))
             .groupby("hour").agg(snr=("snr", "mean"), rssi=("rssi", "mean"),
                                  noise=("noise", "mean"), n=("snr", "size")))
    h = h.join(wxh, how="left")
    h = h.join(time_features(h.index))
    return h.dropna(subset=["temp_c"])


def pdr_hours(pdr: pd.DataFrame, s: str, g: str, wxh: pd.DataFrame) -> pd.DataFrame:
    p = pdr[(pdr["sensor"] == s) & (pdr["gateway"] == g)]
    h = p.groupby("bin").agg(pdr=("pdr", "mean"), expected=("expected", "sum"))
    h.index = pd.DatetimeIndex(h.index, tz="UTC") if h.index.tz is None else h.index
    h = h.join(wxh, how="left").join(time_features(h.index))
    return h.dropna(subset=["temp_c"])


# ==========================================================================
# Estimator B: within-week fixed effects, week-block bootstrap
# ==========================================================================
class FixedEffects:
    """y ~ x + hour-of-day FE + week FE, solved by within-week demeaning.

    Demeaning is done inside each week, so it does not depend on which weeks
    a bootstrap draw picks: the demeaned rows are computed once and the
    bootstrap only re-selects them.
    """

    def __init__(self, h: pd.DataFrame, y: str, xs: list[str]):
        d = h.dropna(subset=[y, *xs])
        self.ok = len(d) >= 200 and d["week"].nunique() >= cfg.MIN_WEEKS
        if not self.ok:
            return
        hours = pd.get_dummies(d["hour_local"], prefix="h", drop_first=True, dtype=float)
        X = np.column_stack([d[xs].to_numpy(float), hours.to_numpy()])
        yv = d[y].to_numpy(float)
        week = d["week"].to_numpy()
        self.Xd = X - pd.DataFrame(X).groupby(week).transform("mean").to_numpy()
        self.yd = yv - pd.Series(yv).groupby(week).transform("mean").to_numpy()
        self.k = len(xs)
        self.n_hours, self.n_weeks = len(d), int(d["week"].nunique())
        weeks = pd.Series(np.arange(len(d))).groupby(week)
        self.week_keys = [k for k, _ in weeks]
        self.rows = [idx.to_numpy() for _, idx in weeks]
        self.key_index = {k: i for i, k in enumerate(self.week_keys)}

    def _solve(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.linalg.lstsq(X, y, rcond=None)[0]

    def beta(self) -> np.ndarray:
        return self._solve(self.Xd, self.yd)[: self.k]

    def boot(self, rng: np.random.Generator, draws: list[np.ndarray] | None = None
             ) -> np.ndarray:
        """Bootstrap betas.  Pass `draws` (week indices) for a paired bootstrap."""
        out = []
        draws = draws or [rng.integers(0, len(self.rows), len(self.rows))
                          for _ in range(cfg.N_BOOT)]
        for pick in draws:
            idx = np.concatenate([self.rows[i] for i in pick])
            out.append(self._solve(self.Xd[idx], self.yd[idx])[: self.k])
        return np.array(out)

    def partial_r2(self, col: int = 0) -> float:
        """Share of the within-week, within-hour variance column `col` explains."""
        full = self.yd - self.Xd @ self._solve(self.Xd, self.yd)
        X0 = np.delete(self.Xd, col, axis=1)
        red = self.yd - X0 @ self._solve(X0, self.yd)
        rss_f, rss_r = float(full @ full), float(red @ red)
        return (rss_r - rss_f) / rss_r if rss_r > 0 else np.nan


def ci(samples: np.ndarray) -> tuple[float, float]:
    s = samples[np.isfinite(samples)]
    return (float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))) if len(s) else (np.nan, np.nan)


# ==========================================================================
# Estimator A: diurnal contrast
# ==========================================================================
def diurnal_contrast(h: pd.DataFrame, y: str, rng: np.random.Generator) -> dict:
    """Per week: day minus night of the metric, against day minus night of T."""
    day = h[h["solar_elev"] > cfg.DAY_SOLAR_ELEV]
    night = h[h["solar_elev"] < cfg.NIGHT_SOLAR_ELEV]

    def side(df):
        return df.groupby("week").agg(m=(y, "mean"), t=("temp_c", "mean"), n=(y, "count"))

    w = side(day).join(side(night), lsuffix="_d", rsuffix="_n", how="inner")
    w = w[(w["n_d"] >= cfg.MIN_HOURS_PER_SIDE) & (w["n_n"] >= cfg.MIN_HOURS_PER_SIDE)].dropna()
    if len(w) < cfg.MIN_WEEKS:
        return {"weeks": len(w)}
    dM, dT = (w["m_d"] - w["m_n"]).to_numpy(), (w["t_d"] - w["t_n"]).to_numpy()
    slope = float(np.polyfit(dT, dM, 1)[0])
    boots = []
    for _ in range(cfg.N_BOOT):
        i = rng.integers(0, len(w), len(w))
        if np.ptp(dT[i]) > 0:
            boots.append(np.polyfit(dT[i], dM[i], 1)[0])

    # Is the day/night gap people rather than heat?  Same contrast on
    # weekdays only and on weekends only: campus activity differs, the sun
    # does not.
    def gap(mask):
        sub = h[mask]
        dd = sub[sub["solar_elev"] > cfg.DAY_SOLAR_ELEV].groupby("week")[y].mean()
        nn = sub[sub["solar_elev"] < cfg.NIGHT_SOLAR_ELEV].groupby("week")[y].mean()
        return (dd - nn).dropna()
    wk, we = gap(h["weekday"]), gap(~h["weekday"])
    common = wk.index.intersection(we.index)
    activity = float((wk[common] - we[common]).mean()) if len(common) >= cfg.MIN_WEEKS else np.nan

    lo, hi = ci(np.array(boots))
    return {"weeks": len(w), "day_minus_night_db": float(dM.mean()),
            "day_minus_night_c": float(dT.mean()), "slope_db_per_c": slope,
            "slope_lo": lo, "slope_hi": hi, "weekday_minus_weekend_db": activity}


# ==========================================================================
# Tail drop
# ==========================================================================
def tail_drop(part: pd.DataFrame, wxh: pd.DataFrame, metric: str) -> dict:
    """p05 of the metric on hot afternoons minus on mild afternoons.

    Same season (May-Sep, so foliage is out on both sides) and the same
    hours (12:00-18:00 local), so neither season nor time of day can make
    the difference.
    """
    part = part[part["sf"] == cfg.ANALYSIS_SF].dropna(subset=[metric])
    hour = part["ts"].dt.floor("1h")
    t = wxh["temp_c"].reindex(hour).to_numpy()
    local = part["ts"].dt.tz_convert(cfg.LOCAL_TZ)
    x = part[metric].to_numpy() - float(part[metric].median())
    lo_h, hi_h = cfg.AFTERNOON_HOURS
    keep = (local.dt.month.isin(cfg.SUMMER_MONTHS).to_numpy()
            & local.dt.hour.between(lo_h, hi_h).to_numpy())
    hot = x[keep & (t >= cfg.HOT_C)]
    mild = x[keep & (t >= cfg.MILD_C[0]) & (t <= cfg.MILD_C[1])]
    if len(hot) < cfg.MIN_TAIL_SAMPLES or len(mild) < cfg.MIN_TAIL_SAMPLES:
        return {"n_hot": len(hot), "n_mild": len(mild)}
    return {"n_hot": len(hot), "n_mild": len(mild),
            "p05_hot_minus_mild_db": float(np.quantile(hot, 0.05) - np.quantile(mild, 0.05)),
            "median_hot_minus_mild_db": float(np.median(hot) - np.median(mild))}


# ==========================================================================
# Step changes
# ==========================================================================
def find_steps(part: pd.DataFrame, metric: str, wxh: pd.DataFrame) -> list[dict]:
    """Abrupt, persistent level shifts in a link's daily median.

    A step must be large over one week AND persist over four, which a
    seasonal climate drift (a fraction of a dB per week) cannot satisfy.
    Steps are reported, not removed: the week fixed effects already absorb
    them for this analysis.  They matter for Week 3's forecasting splits.
    """
    part = part[part["sf"] == cfg.ANALYSIS_SF]
    day = part.set_index("ts")[metric].resample("1D").median()
    day = day.reindex(pd.date_range(day.index.min(), day.index.max(), freq="1D"))
    temp = wxh["temp_c"].resample("1D").mean().reindex(day.index)

    def pre(x, n):
        return x.rolling(n, min_periods=n // 2 + 1).median().shift(1)

    def post(x, n):
        return x[::-1].rolling(n, min_periods=n // 2 + 1).median()[::-1]

    s7 = post(day, cfg.STEP_SHORT_DAYS) - pre(day, cfg.STEP_SHORT_DAYS)
    s28 = post(day, cfg.STEP_LONG_DAYS) - pre(day, cfg.STEP_LONG_DAYS)
    t28 = post(temp, cfg.STEP_LONG_DAYS) - pre(temp, cfg.STEP_LONG_DAYS)
    hit = ((s7.abs() >= cfg.STEP_SHORT_DB) & (s28.abs() >= cfg.STEP_LONG_DB)
           & (np.sign(s7) == np.sign(s28)))
    cands = s7[hit].abs().sort_values(ascending=False)
    steps, taken = [], []
    for when in cands.index:
        if any(abs((when - t).days) < 14 for t in taken):
            continue
        taken.append(when)
        steps.append({"date": str(when.date()), "jump_7d_db": round(float(s7[when]), 2),
                      "jump_28d_db": round(float(s28[when]), 2),
                      "temp_change_28d_c": round(float(t28.get(when, np.nan)), 1)})
    return sorted(steps, key=lambda r: r["date"])


# ==========================================================================
# Main analysis
# ==========================================================================
def analyse(rec: pd.DataFrame, pdr: pd.DataFrame, wx: pd.DataFrame) -> dict:
    rng = np.random.default_rng(cfg.BOOT_SEED)
    wxh = weather_hourly(wx)
    t_all = wxh["temp_c"].dropna()
    t_typ, t_hot = float(t_all.median()), float(t_all.quantile(0.95))
    swing_c = t_hot - t_typ

    tiers = link_tiers(rec, pdr)
    groups = dict(tuple(rec.groupby(["sensor", "gateway"], sort=False)))

    effects, diurnal, tails, pdr_rows, steps = [], [], [], [], []
    fits: dict[tuple, dict] = {}       # kept for the natural experiment

    for row in tiers.itertuples():
        key = (row.sensor, row.gateway)
        if row.tier == "excluded":
            continue
        part = groups[key]
        h = link_hours(part, wxh, row.snr_ceiling)
        fits[key] = {"hours": h}

        # Fixed effects on SNR, RSSI and the implied noise floor.  Comparing
        # the three on one link says WHERE a temperature effect sits: in the
        # path (SNR and RSSI move, noise does not) or in the gateway's
        # measurement (RSSI and noise move together, SNR does not).
        for metric in ("snr", "rssi", "noise"):
            fe = FixedEffects(h, metric, ["temp_c", "rh_pct"])
            if not fe.ok:
                continue
            fits[key][metric] = fe
            b = fe.beta()
            bs = fe.boot(rng)
            lo, hi = ci(bs[:, 0])
            lag = {}
            for lag_col in ("temp_lag1", "temp_lag3"):
                fl = FixedEffects(h, metric, [lag_col, "rh_pct"])
                lag[lag_col] = float(fl.beta()[0]) if fl.ok else np.nan
            ext_cols = [c for c in ("pressure_hpa", "rain_mm") if c in h.columns]
            fx = FixedEffects(h, metric, ["temp_c", "rh_pct", *ext_cols])
            effects.append({
                "sensor": row.sensor, "gateway": row.gateway, "tier": row.tier,
                "metric": metric, "primary": metric == row.metric,
                "hours": fe.n_hours, "weeks": fe.n_weeks,
                "db_per_c": float(b[0]), "db_per_c_lo": lo, "db_per_c_hi": hi,
                "hot_swing_db": float(b[0]) * swing_c,
                "hot_swing_lo": lo * swing_c, "hot_swing_hi": hi * swing_c,
                "db_per_10pp_rh": float(b[1]) * 10,
                "partial_r2_temp": fe.partial_r2(0),
                "db_per_c_lag1": lag["temp_lag1"], "db_per_c_lag3": lag["temp_lag3"],
                "db_per_c_with_pressure_rain": float(fx.beta()[0]) if fx.ok else np.nan,
            })
            if metric == "noise":
                continue
            d = diurnal_contrast(h, metric, rng)
            diurnal.append({"sensor": row.sensor, "gateway": row.gateway,
                            "tier": row.tier, "metric": metric, **d})
            tails.append({"sensor": row.sensor, "gateway": row.gateway,
                          "tier": row.tier, "metric": metric,
                          **tail_drop(part, wxh, metric)})

        # PDR on every analysable link; it is the primary metric on the
        # marginal and floor tiers, where SNR is censored.
        ph = pdr_hours(pdr, row.sensor, row.gateway, wxh)
        fits[key]["pdr_hours"] = ph
        fe_p = FixedEffects(ph, "pdr", ["temp_c", "rh_pct"])
        if fe_p.ok:
            b = fe_p.beta()
            lo, hi = ci(fe_p.boot(rng)[:, 0])
            pdr_rows.append({"sensor": row.sensor, "gateway": row.gateway,
                             "tier": row.tier, "primary": row.metric == "pdr",
                             "mean_pdr": float(ph["pdr"].mean()),
                             "pp_per_c": float(b[0]) * 100,
                             "hot_swing_pp": float(b[0]) * swing_c * 100,
                             "hot_swing_lo_pp": lo * swing_c * 100,
                             "hot_swing_hi_pp": hi * swing_c * 100,
                             "weeks": fe_p.n_weeks})
            fits[key]["pdr"] = fe_p

        for step_metric in ("snr", "rssi"):
            for st in find_steps(part, step_metric, wxh):
                steps.append({"sensor": row.sensor, "gateway": row.gateway,
                              "metric": step_metric, **st})

    natural = natural_experiment(tiers, fits, rng, swing_c)
    steps = pd.DataFrame(steps)
    gw_summary = gateway_summary(tiers, fits, rng, swing_c)
    gate = pooled([fits[(r.sensor, r.gateway)]["snr"] for r in tiers.itertuples()
                   if r.tier in ("mid_range", "near_ceiling")
                   and "snr" in fits.get((r.sensor, r.gateway), {})], rng, swing_c)

    return {"tiers": tiers, "effects": pd.DataFrame(effects),
            "diurnal": pd.DataFrame(diurnal), "tails": pd.DataFrame(tails),
            "pdr": pd.DataFrame(pdr_rows), "steps": steps,
            "gateway_events": gateway_events(steps),
            "gateways": gw_summary, "gate_pooled": gate,
            "profile": temperature_profile(tiers, fits),
            "natural": natural, "t_typ": t_typ, "t_hot": t_hot, "swing_c": swing_c}


# ==========================================================================
# Pooled estimates, per gateway
# ==========================================================================
def pooled(fes: list, rng: np.random.Generator, scale: float) -> dict:
    """Mean temperature slope across links, with a joint week bootstrap.

    Every link in a draw uses the SAME calendar weeks, so weather shared by
    all links is resampled once, and the interval reflects that the links
    are not independent of each other.
    """
    fes = [f for f in fes if f is not None and f.ok]
    if not fes:
        return {"links": 0, "swing": np.nan, "lo": np.nan, "hi": np.nan}
    point = float(np.mean([f.beta()[0] for f in fes])) * scale
    union = sorted(set().union(*[f.week_keys for f in fes]))
    boots = []
    for _ in range(cfg.N_BOOT):
        pick = rng.choice(union, len(union))
        vals = []
        for f in fes:
            idx = [f.key_index[w] for w in pick if w in f.key_index]
            if len(idx) >= cfg.MIN_WEEKS:
                vals.append(f.boot(rng, [np.array(idx)])[0, 0])
        if vals:
            boots.append(np.mean(vals) * scale)
    lo, hi = ci(np.array(boots))
    return {"links": len(fes), "swing": point, "lo": lo, "hi": hi}


def gateway_summary(tiers: pd.DataFrame, fits: dict, rng: np.random.Generator,
                    swing_c: float) -> pd.DataFrame:
    """Where does the effect live?  One row per gateway, four metrics pooled.

    SNR and noise use only links where SNR is neither pinned at the ceiling
    nor at the floor most of the time; RSSI and PDR use every analysable link.
    """
    snr_ok = {"mid_range", "near_ceiling", "marginal"}
    rows = []
    for g in sorted(cfg.GATEWAYS):
        here = [r for r in tiers.itertuples() if r.gateway == g and r.tier != "excluded"]
        def get(metric, allowed=None):
            return [fits.get((r.sensor, r.gateway), {}).get(metric) for r in here
                    if allowed is None or r.tier in allowed]
        row = {"gateway": g, "placement": cfg.GATEWAYS[g]["placement"],
               "indoor": cfg.GATEWAYS[g]["indoor"]}
        for metric, allowed, scale, unit in (("snr", snr_ok, swing_c, "db"),
                                             ("rssi", None, swing_c, "db"),
                                             ("noise", snr_ok, swing_c, "db"),
                                             ("pdr", None, swing_c * 100, "pp")):
            r = pooled(get(metric, allowed), rng, scale)
            row[f"{metric}_links"] = r["links"]
            row[f"{metric}_{unit}"] = r["swing"]
            row[f"{metric}_lo"] = r["lo"]
            row[f"{metric}_hi"] = r["hi"]
        noise_levels = [fits[(r.sensor, r.gateway)]["hours"]["noise"].median()
                        for r in here if r.tier in snr_ok and (r.sensor, r.gateway) in fits]
        row["noise_floor_dbm"] = float(np.nanmedian(noise_levels)) if noise_levels else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def gateway_events(steps: pd.DataFrame, window_days: int = 3, min_sensors: int = 3
                   ) -> pd.DataFrame:
    """Steps that hit several sensors on one gateway at once.

    A real change in one link's path moves one link.  The same jump on three
    or more sensors within a few days, through one gateway, is something
    that happened AT the gateway: a reboot, a firmware change, a
    recalibration, an antenna or cable fault.
    """
    if steps.empty:
        return pd.DataFrame()
    st = steps.assign(date=pd.to_datetime(steps["date"]))
    out, used = [], set()
    for (g, m), grp in st.groupby(["gateway", "metric"]):
        grp = grp.sort_values("date")
        for when in grp["date"]:
            if any(abs((when - u).days) <= 7 for (gg, mm, u) in used if gg == g and mm == m):
                continue
            near = grp[(grp["date"] - when).abs() <= pd.Timedelta(days=window_days)]
            if near["sensor"].nunique() >= min_sensors:
                used.add((g, m, when))
                out.append({"date": str(when.date()), "gateway": g, "metric": m,
                            "sensors": near["sensor"].nunique(),
                            "median_jump_db": float(near["jump_7d_db"].median()),
                            "temp_change_28d_c": float(near["temp_change_28d_c"].median())})
    return pd.DataFrame(out).sort_values(["gateway", "date"]) if out else pd.DataFrame()


def temperature_profile(tiers: pd.DataFrame, fits: dict, width: float = 3.0) -> pd.DataFrame:
    """Median deviation from each link's own weekly mean, by temperature bin.

    Descriptive only (no hour-of-day control): it shows the SHAPE of the
    relationship per gateway -- a straight line, a threshold, or nothing.
    """
    rows = []
    for r in tiers.itertuples():
        f = fits.get((r.sensor, r.gateway))
        if r.tier == "excluded" or f is None:
            continue
        h = f["hours"]
        for metric in ("snr", "rssi"):
            if metric == "snr" and r.tier not in ("mid_range", "near_ceiling", "marginal"):
                continue
            dev = h[metric] - h.groupby("week")[metric].transform("mean")
            rows.append(pd.DataFrame({"gateway": r.gateway, "metric": metric,
                                      "bin": np.floor(h["temp_c"] / width) * width,
                                      "dev": dev}))
    if not rows:
        return pd.DataFrame()
    allr = pd.concat(rows).dropna()
    return (allr.groupby(["gateway", "metric", "bin"])["dev"]
                .agg(median="median", hours="size").reset_index())


def natural_experiment(tiers: pd.DataFrame, fits: dict, rng: np.random.Generator,
                       swing_c: float) -> pd.DataFrame:
    """Outdoor-minus-indoor temperature slope for each co-located pair.

    Both links are fitted on the same bootstrap draw of weeks (paired), so
    the interval is on the DIFFERENCE, not two separate intervals.  The
    replicate pair (01/07, both indoor, same spot) gives the difference two
    identical placements produce by chance -- the bar to clear.
    """
    tier_of = {(r.sensor, r.gateway): r for r in tiers.itertuples()}
    rows = []
    for kind, pairs in (("contrast", cfg.CONTRAST_PAIRS), ("replicate", cfg.REPLICATE_PAIRS)):
        for a, b in pairs:
            for g in sorted(cfg.GATEWAYS):
                ta, tb = tier_of.get((a, g)), tier_of.get((b, g))
                if ta is None or tb is None or "excluded" in (ta.tier, tb.tier):
                    continue
                # Both links must be judged on the same metric: the weakest
                # one either link allows (censored -> PDR, saturated -> RSSI).
                metrics = {ta.metric, tb.metric}
                metric = "pdr" if "pdr" in metrics else ("rssi" if "rssi" in metrics else "snr")
                table = "pdr_hours" if metric == "pdr" else "hours"
                ha = fits.get((a, g), {}).get(table)
                hb = fits.get((b, g), {}).get(table)
                if ha is None or hb is None:
                    continue
                fea = FixedEffects(ha, metric, ["temp_c", "rh_pct"])
                feb = FixedEffects(hb, metric, ["temp_c", "rh_pct"])
                if not (fea.ok and feb.ok):
                    continue
                scale = swing_c * (100.0 if metric == "pdr" else 1.0)   # pp or dB
                # Pair weeks by label so each draw takes the same weeks on both.
                wa = sorted(ha.dropna(subset=[metric, "temp_c", "rh_pct"])["week"].unique())
                wb = sorted(hb.dropna(subset=[metric, "temp_c", "rh_pct"])["week"].unique())
                common = sorted(set(wa) & set(wb))
                if len(common) < cfg.MIN_WEEKS:
                    continue
                ia = {w: i for i, w in enumerate(wa)}
                ib = {w: i for i, w in enumerate(wb)}
                draws = [rng.choice(common, len(common)) for _ in range(cfg.N_BOOT)]
                ba = fea.boot(rng, [np.array([ia[w] for w in d]) for d in draws])[:, 0]
                bb = feb.boot(rng, [np.array([ib[w] for w in d]) for d in draws])[:, 0]
                lo, hi = ci((ba - bb) * scale)
                rows.append({
                    "pair": f"{a[-2:]}/{b[-2:]}", "kind": kind, "gateway": g, "metric": metric,
                    "unit": "pp" if metric == "pdr" else "dB",
                    "first_swing": float(fea.beta()[0]) * scale,
                    "second_swing": float(feb.beta()[0]) * scale,
                    "difference": float(fea.beta()[0] - feb.beta()[0]) * scale,
                    "difference_lo": lo, "difference_hi": hi, "common_weeks": len(common),
                })
    return pd.DataFrame(rows)


# ==========================================================================
# Verdict and report
# ==========================================================================
def verdict(res: dict) -> dict:
    e = res["effects"]
    gate = e[(e["primary"]) & (e["metric"] == "snr")
             & (e["tier"].isin(["mid_range", "near_ceiling"]))]
    if gate.empty:
        return {"label": "No verdict", "median_swing_db": np.nan, "links": 0}
    med = float(gate["hot_swing_db"].median())
    # The verdict rests on the POOLED estimate, whose interval accounts for
    # all links sharing the same weather; the per-link median is reported
    # beside it.  The interval's far end is checked too: a small point
    # estimate with a wide interval is "not shown", not "small".
    pool = res.get("gate_pooled") or {}
    size = abs(pool.get("swing", med)) if np.isfinite(pool.get("swing", np.nan)) else abs(med)
    far = max(abs(pool.get("lo", np.nan)), abs(pool.get("hi", np.nan))) \
        if np.isfinite(pool.get("lo", np.nan)) else np.nan
    if size >= cfg.GATE_LEADS_DB:
        label, headline = "Climate leads", "Climate-aware ADR, as planned"
    elif size >= cfg.GATE_MODULATES_DB:
        label, headline = ("Climate modulates the margin",
                           "Climate-conditioned reliability margin plus SF/multi-gateway allocation")
    else:
        label, headline = ("Climate is a minor guard",
                           "Multi-gateway diversity and collision-aware allocation; "
                           "report the small climate effect honestly")
    excl0 = int(((gate["hot_swing_hi"] < 0) | (gate["hot_swing_lo"] > 0)).sum())
    ruled_out = "unknown"
    if np.isfinite(far):
        ruled_out = ("an effect above {:.1f} dB is ruled out".format(cfg.GATE_MODULATES_DB)
                     if far < cfg.GATE_MODULATES_DB else
                     "an effect up to {:.2f} dB cannot be ruled out".format(far))
    return {"label": label, "headline": headline, "median_swing_db": med,
            "pooled_swing_db": pool.get("swing", np.nan),
            "pooled_lo": pool.get("lo", np.nan), "pooled_hi": pool.get("hi", np.nan),
            "uncertainty": ruled_out,
            "min_swing_db": float(gate["hot_swing_db"].min()),
            "max_swing_db": float(gate["hot_swing_db"].max()),
            "links": len(gate), "links_ci_excludes_zero": excl0,
            "negative_links": int((gate["hot_swing_db"] < 0).sum())}


def md_table(df: pd.DataFrame, cols: list[str], names: list[str] | None = None,
             fmt: str | dict = "{:+.2f}") -> str:
    """Pipe table.  `fmt` is one format for every float, or {column: format}."""
    names = names or cols
    df = df.copy()
    for c in cols:                       # a column the analysis never filled
        if c not in df.columns:
            df[c] = np.nan
    out = ["| " + " | ".join(names) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in df[cols].itertuples(index=False):
        cells = []
        for c, v in zip(cols, r):
            f = fmt.get(c, "{:+.2f}") if isinstance(fmt, dict) else fmt
            if isinstance(v, (float, np.floating)):
                cells.append("—" if not np.isfinite(v) else f.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def gateway_lines(res: dict) -> list[str]:
    g = res.get("gateways")
    if g is None or g.empty:
        return ["Not available."]
    t = g.copy()
    t["gw"] = t["gateway"].str[-1] + np.where(t["indoor"], " (indoor)", " (roof)")
    def band(m, u):
        return [f"{a:+.2f} [{b:+.2f}, {c:+.2f}]" if np.isfinite(a) else "—"
                for a, b, c in zip(t[f"{m}_{u}"], t[f"{m}_lo"], t[f"{m}_hi"])]
    t["SNR dB"] = band("snr", "db")
    t["RSSI dB"] = band("rssi", "db")
    t["noise floor dB"] = band("noise", "db")
    t["PDR pp"] = band("pdr", "pp")
    t["floor dBm"] = t["noise_floor_dbm"]
    t["links"] = t["rssi_links"]
    return [md_table(t, ["gw", "links", "SNR dB", "RSSI dB", "noise floor dB", "PDR pp",
                         "floor dBm"], fmt="{:.1f}")]


def report(res: dict, v: dict) -> str:
    e, d, t, p, n, s = (res[k] for k in ("effects", "diurnal", "tails", "pdr", "natural", "steps"))
    sw = res["swing_c"]
    L = [f"# Week 2 — climate gate",
         "",
         f"**Verdict: {v['label']}.** A hot afternoon ({res['t_hot']:.1f} °C, the 95th "
         f"percentile) against a typical hour ({res['t_typ']:.1f} °C) moves SNR by "
         f"**{v.get('pooled_swing_db', np.nan):+.2f} dB** pooled over the {v['links']} links "
         f"where SNR can move (95% interval {v.get('pooled_lo', np.nan):+.2f} to "
         f"{v.get('pooled_hi', np.nan):+.2f} dB: {v.get('uncertainty', '')}). Per-link "
         f"median {v['median_swing_db']:+.2f} dB, range {v['min_swing_db']:+.2f} to "
         f"{v['max_swing_db']:+.2f} dB. Headline: {v['headline']}.",
         "",
         f"Gate yardstick: one SF step is about 2.5 dB, one TP step 3 dB. Thresholds: "
         f"{cfg.GATE_LEADS_DB} dB leads, {cfg.GATE_MODULATES_DB} dB modulates. CASM's "
         f"free-space SF7 prior was {cfg.CASM_PRIOR_SF7_DB} dB from a median split with no "
         "time-of-day control, so it is an upper bound on what this estimator should find.",
         "",
         "## How the number was made",
         "",
         "Each link is compared only with itself, on SF7. Week fixed effects remove season, "
         "foliage and step changes; hour-of-day fixed effects remove the campus's daily "
         "rhythm. Temperature is credited only when an hour runs hotter or cooler than that "
         "hour usually does that week. Intervals are 95%, from "
         f"{cfg.N_BOOT} bootstrap draws of whole weeks.",
         "",
         "## Where the effect lives: per gateway",
         "",
         "Pooled over each gateway's links, one shared week bootstrap. Propagation loss "
         "moves SNR and RSSI together and leaves the noise floor alone. A change in the "
         "gateway's gain or RSSI calibration moves RSSI and the noise floor together and "
         "leaves SNR alone. Thermal noise at 125 kHz cannot fall below -123 dBm, and "
         "rises (by about 0.2 dB) with heat rather than falling.",
         "",
         *gateway_lines(res),
         "",
         "## Signal links (SNR and RSSI, fixed effects)",
         "",
         f"`swing` = effect of +{sw:.1f} °C. `diurnal` is estimator A: the day-minus-night "
         "slope across weeks, scaled the same way. The two should agree in sign.",
         ""]
    for frame, cols in ((d, ["slope_db_per_c", "weekday_minus_weekend_db"]),
                        (t, ["p05_hot_minus_mild_db"])):
        for c in cols:
            if c not in frame.columns:
                frame[c] = np.nan
    if not e.empty:
        merged = e.merge(d[["sensor", "gateway", "metric", "slope_db_per_c",
                            "weekday_minus_weekend_db"]],
                         on=["sensor", "gateway", "metric"], how="left")
        merged = merged.merge(t[["sensor", "gateway", "metric", "p05_hot_minus_mild_db"]],
                              on=["sensor", "gateway", "metric"], how="left")
        merged["link"] = merged["sensor"].str[-2:] + "→" + merged["gateway"].str[-1]
        merged["diurnal_swing"] = merged["slope_db_per_c"] * sw
        merged["primary"] = np.where(merged["primary"], "yes", "")
        merged = merged.sort_values(["metric", "tier", "link"])
        L += [md_table(merged, ["link", "tier", "metric", "primary", "hot_swing_db",
                                "hot_swing_lo", "hot_swing_hi", "diurnal_swing",
                                "p05_hot_minus_mild_db", "db_per_10pp_rh",
                                "partial_r2_temp", "weekday_minus_weekend_db"],
                       ["link", "tier", "metric", "primary", "swing dB", "lo", "hi",
                        "diurnal dB", "p05 hot−mild", "per +10pp RH", "partial R²",
                        "weekday−weekend"],
                       fmt={"partial_r2_temp": "{:.3f}"}),
              "",
              "`p05 hot−mild` compares the worst 5% of receptions on hot (≥30 °C) and mild "
              "(18–25 °C) afternoons, May–September, 12:00–18:00 — same season, same hours. "
              "`weekday−weekend` is the day/night gap on weekdays minus on weekends: a large "
              "value means people, not heat, drive part of the daily cycle.",
              ""]
        lag = e[e["primary"]][["sensor", "gateway", "metric", "db_per_c", "db_per_c_lag1",
                               "db_per_c_lag3", "db_per_c_with_pressure_rain"]].copy()
        lag["link"] = lag["sensor"].str[-2:] + "→" + lag["gateway"].str[-1]
        L += ["### Robustness: lagged temperature, pressure and rain", "",
              md_table(lag, ["link", "metric", "db_per_c", "db_per_c_lag1", "db_per_c_lag3",
                             "db_per_c_with_pressure_rain"],
                       ["link", "metric", "dB/°C now", "1 h earlier", "3 h earlier",
                        "with pressure + rain"], fmt="{:+.3f}"), ""]

    if not p.empty:
        pp = p.copy()
        pp["link"] = pp["sensor"].str[-2:] + "→" + pp["gateway"].str[-1]
        pp["primary"] = np.where(pp["primary"], "yes", "")
        pp = pp.sort_values(["primary", "tier", "link"], ascending=[False, True, True])
        L += ["## Packet delivery (the censored links)", "",
              "On marginal and floor links SNR is only logged for packets that got through, "
              "so heat shows up here, as delivery lost, rather than as a lower SNR.", "",
              md_table(pp, ["link", "tier", "primary", "mean_pdr", "hot_swing_pp",
                            "hot_swing_lo_pp", "hot_swing_hi_pp"],
                       ["link", "tier", "primary", "mean PDR", f"swing pp (+{sw:.0f} °C)",
                        "lo", "hi"], fmt={"mean_pdr": "{:.2f}", "hot_swing_pp": "{:+.1f}",
                                          "hot_swing_lo_pp": "{:+.1f}",
                                          "hot_swing_hi_pp": "{:+.1f}"}), ""]

    if not n.empty:
        L += ["## Natural experiment", "",
              "First sensor of each contrast pair is outdoors, second indoors. If the "
              "outdoor node's swing is clearly larger, the effect lives in the node's "
              "hardware; if they match, it lives in the path or the gateway. The replicate "
              "pair's difference is what two identical placements give by chance. Both links "
              "of a pair use the metric the weaker link allows; `pp` rows are PDR.", "",
              md_table(n, ["pair", "kind", "gateway", "metric", "unit", "first_swing",
                           "second_swing", "difference", "difference_lo", "difference_hi",
                           "common_weeks"],
                       ["pair", "kind", "gw", "metric", "unit", "first", "second",
                        "difference", "lo", "hi", "weeks"]), ""]

    L += ["## Step changes", ""]
    if s.empty:
        L += ["None found.", ""]
    else:
        ss = s.copy()
        ss["link"] = ss["sensor"].str[-2:] + "→" + ss["gateway"].str[-1]
        L += ["Abrupt, lasting level shifts. The week fixed effects already absorb them "
              "here; Week 3's time splits must not straddle them. A large temperature "
              "change in the same window means the 'step' may be weather after all.", "",
              md_table(ss, ["date", "link", "metric", "jump_7d_db", "jump_28d_db",
                            "temp_change_28d_c"],
                       ["date", "link", "metric", "7-day jump dB", "28-day jump dB",
                        "28-day ΔT °C"]), ""]

    ev = res.get("gateway_events")
    if ev is not None and not ev.empty:
        L += ["### Gateway events", "",
              "The same jump on three or more sensors through one gateway within three "
              "days happened at the gateway (reboot, firmware, recalibration, antenna), "
              "not in any one path.", "",
              md_table(ev, ["date", "gateway", "metric", "sensors", "median_jump_db",
                            "temp_change_28d_c"],
                       ["date", "gateway", "metric", "sensors", "median 7-day jump dB",
                        "28-day ΔT °C"], fmt="{:+.1f}"), ""]

    prof = res.get("profile")
    if prof is not None and not prof.empty:
        pv = prof[prof["hours"] >= 50].pivot_table(index="bin", columns=["gateway", "metric"],
                                                   values="median")
        pv.columns = [f"{g[-1]} {m}" for g, m in pv.columns]
        pv = pv.reset_index()
        pv["bin"] = pv["bin"].map(lambda b: f"{b:.0f} to {b + 3:.0f}")
        cols = list(pv.columns)
        L += ["## Shape of the relationship", "",
              "Median deviation (dB) from each link's own weekly mean, by temperature, "
              "pooled per gateway. No hour-of-day control, so this shows shape, not size: "
              "a straight line, a threshold, or nothing.", "",
              md_table(pv, cols, ["°C"] + cols[1:]), ""]

    tiers = res["tiers"].copy()
    tiers["link"] = tiers["sensor"].str[-2:] + "→" + tiers["gateway"].str[-1]
    tiers["metric"] = tiers["metric"].fillna("—")
    L += ["## Link tiers (decided from the data)", "",
          md_table(tiers.sort_values(["tier", "link"]),
                   ["link", "tier", "metric", "rx", "snr_mean", "snr_p05", "at_ceiling", "pdr"],
                   ["link", "tier", "metric", "rx", "SNR mean", "SNR p05", "at ceiling", "PDR"],
                   fmt={"snr_mean": "{:.1f}", "snr_p05": "{:.1f}", "at_ceiling": "{:.1%}",
                        "pdr": "{:.2f}"}), ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="./build")
    ap.add_argument("--out", default="./week2")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("loading ...")
    rec, pdr, wx = load(Path(args.build))
    print(f"  {len(rec):,} receptions, {len(pdr):,} PDR rows")
    print("analysing (bootstrap: a few minutes) ...")
    res = analyse(rec, pdr, wx)
    v = verdict(res)

    for name in ("tiers", "effects", "diurnal", "tails", "pdr", "steps", "natural",
                 "gateways", "gateway_events", "profile"):
        res[name].to_csv(out / f"{name}.csv", index=False)
    (out / "verdict.json").write_text(json.dumps(v, indent=2, default=float))
    (out / "CLIMATE_GATE.md").write_text(report(res, v))

    print(f"\nVERDICT: {v['label']}")
    if v.get("links"):
        print(f"  pooled hot-afternoon SNR swing {v['pooled_swing_db']:+.2f} dB "
              f"[{v['pooled_lo']:+.2f}, {v['pooled_hi']:+.2f}] over {v['links']} links: "
              f"{v['uncertainty']}")
        gw = res["gateways"]
        for r in gw.itertuples():
            print(f"  {r.gateway}: SNR {r.snr_db:+.2f}  RSSI {r.rssi_db:+.2f}  "
                  f"noise floor {r.noise_db:+.2f} dB  (median floor {r.noise_floor_dbm:.1f} dBm)")
    print(f"wrote {out}/CLIMATE_GATE.md and the CSVs beside it")


if __name__ == "__main__":
    main()
