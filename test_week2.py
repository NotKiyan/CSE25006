"""Known-answer tests for the Week 2 estimators.

READ THIS FIRST.  Like test_pipeline.py, the fixture here is NOT data and
never becomes a result.  It plants an effect of known size together with
every confounder the analysis claims to remove, and checks that the
estimators recover the planted number and nothing else:

  * planted:     SNR falls 0.10 dB per °C on link 02->A; RSSI 0.10 dB per °C
                 on the saturated link; PDR 1 pp per °C on the censored link.
  * confounders: a 3 dB summer foliage drop that tracks the season's heat,
                 a 1.5 dB weekday-daytime "people" effect, a +4 dB step on
                 2024-07-15, and SNR pinned at a 14.5 dB reporting ceiling.
  * null:        link 03->A shares every confounder but has NO temperature
                 effect.  Its estimate must straddle zero.

No number from this file appears in any result.

    python test_week2.py        (about a minute: it runs the bootstrap)
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import uva_config as cfg
import uva_week2 as w2

BETA = -0.10           # dB per °C, planted
PDR_BETA = -0.01       # PDR per °C, planted (1 pp)
STEP_DATE = pd.Timestamp("2024-07-15", tz="UTC")


def build_fixture(root: Path) -> None:
    rng = np.random.default_rng(0)
    hours = pd.date_range("2024-02-09", "2024-12-31 23:00", freq="1h", tz="UTC")
    local = hours.tz_convert(cfg.LOCAL_TZ)
    doy = local.dayofyear.to_numpy()
    hod = local.hour.to_numpy() + local.minute.to_numpy() / 60

    # Weather: season + a daily cycle whose size changes week to week
    # (sunny vs cloudy weeks) + slow weather fronts.
    week_id = (local.isocalendar()["week"].astype(int).to_numpy())
    amp = 6 + 3 * rng.standard_normal(60)[week_id % 60]
    front = pd.Series(rng.standard_normal(len(hours))).rolling(36, min_periods=1).mean().to_numpy() * 8
    season = 15 - 13 * np.cos((doy - 20) / 365 * 2 * np.pi)
    temp = season + amp * np.sin((hod - 9) / 24 * 2 * np.pi) + front
    rh = 65 - 1.0 * (temp - 15) + 5 * rng.standard_normal(len(hours))

    foliage = -3.0 * np.clip(np.sin((doy - 100) / 180 * np.pi), 0, None)  # summer only
    weekday_day = np.asarray((local.dayofweek < 5) & (local.hour >= 8) & (local.hour < 18))
    people = -1.5 * weekday_day
    step = np.where(hours >= STEP_DATE, 4.0, 0.0)

    wx = pd.DataFrame({"ts": hours, "temp_c": temp, "rh_pct": rh,
                       "pressure_hpa": 1013 + 5 * rng.standard_normal(len(hours)),
                       "rain_mm": 0.0, "weather_missing": False})

    rows, pdr_rows = [], []
    uid = 0

    def add(sensor, gw, snr_h, rssi_h, keep=None):
        nonlocal uid
        for k in range(6):                                  # one uplink every 10 min
            ts = hours + pd.Timedelta(minutes=10 * k)
            snr = snr_h + 1.5 * rng.standard_normal(len(hours))
            rssi = rssi_h + 1.5 * rng.standard_normal(len(hours))
            m = np.ones(len(hours), bool) if keep is None else keep(snr)
            n = int(m.sum())
            rows.append(pd.DataFrame({
                "ts": ts[m], "sensor": sensor, "gateway": gw, "snr": snr[m],
                "rssi": rssi[m], "sf": 7, "freq": 904_100_000,
                "uplink_id": np.arange(uid, uid + n)}))
            uid += n

    t = temp
    add("sensor02", "gatewayA", 10 + BETA * t + foliage + people + step, -90 + BETA * t + foliage)
    add("sensor03", "gatewayA", 10 + foliage + people + step, -90 + foliage)
    # Gateway artefact: RSSI falls with heat, SNR does not.  The receiver's
    # gain or RSSI calibration drifts; the signal itself is unchanged.
    add("sensor04", "gatewayB", np.full(len(hours), 6.0), -100 + BETA * t)
    # Saturated: SNR pinned at 14.5 dB, so the effect is only visible in RSSI.
    add("sensor02", "gatewayC", np.full(len(hours), 30.0), -80 + BETA * t + foliage)
    # Censored: receptions below the SF7 floor are never logged.
    add("sensor05", "gatewayC", -2 + BETA * t + foliage, -115 + BETA * t,
        keep=lambda s: s >= cfg.SF_FLOOR_DB[7])
    rec = pd.concat(rows, ignore_index=True)
    rec.loc[(rec["sensor"] == "sensor02") & (rec["gateway"] == "gatewayC"), "snr"] = 14.5

    for s, g, base, beta in (("sensor02", "gatewayA", 0.95, 0.0), ("sensor03", "gatewayA", 0.95, 0.0),
                             ("sensor02", "gatewayC", 0.97, 0.0),
                             ("sensor04", "gatewayB", 0.95, 0.0),
                             ("sensor05", "gatewayC", 0.80, PDR_BETA)):
        p = np.clip(base + beta * (temp - 15) + 0.03 * rng.standard_normal(len(hours)), 0, 1)
        pdr_rows.append(pd.DataFrame({"sensor": s, "gateway": g, "bin": hours, "epoch": 0,
                                      "expected": 6, "pdr": p}))
    pdr = pd.concat(pdr_rows, ignore_index=True)

    root.mkdir(parents=True, exist_ok=True)
    rec.to_parquet(root / "uva_receptions.parquet", index=False)
    pdr.to_parquet(root / "uva_pdr.parquet", index=False)
    wx.to_parquet(root / "uva_weather.parquet", index=False)
    (root / "build_stats.json").write_text(json.dumps({"pipeline_version": cfg.PIPELINE_VERSION}))


def run() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        build_fixture(tmp)
        rec, pdr, wx = w2.load(tmp)
        res = w2.analyse(rec, pdr, wx)
        v = w2.verdict(res)
        sw = res["swing_c"]

        tiers = res["tiers"].set_index(["sensor", "gateway"])["tier"]
        assert tiers[("sensor02", "gatewayA")] == "mid_range", tiers
        assert tiers[("sensor03", "gatewayA")] == "mid_range", tiers
        assert tiers[("sensor02", "gatewayC")] == "saturated", tiers
        assert tiers[("sensor05", "gatewayC")] == "marginal", tiers
        print("  tiers        ok   (mid, mid, saturated, marginal — decided from the data)")

        e = res["effects"].set_index(["sensor", "gateway", "metric"])
        b = e.loc[("sensor02", "gatewayA", "snr")]
        assert abs(b["db_per_c"] - BETA) < 0.02, b["db_per_c"]
        assert b["db_per_c_hi"] < 0, "planted effect must be distinguishable from zero"
        n = e.loc[("sensor03", "gatewayA", "snr")]
        assert n["db_per_c_lo"] < 0 < n["db_per_c_hi"], (n["db_per_c_lo"], n["db_per_c_hi"])
        print(f"  planted      ok   ({b['db_per_c']:+.3f} dB/°C recovered vs {BETA:+.2f} planted, "
              f"despite foliage, people and a 4 dB step)")
        print(f"  null         ok   ({n['db_per_c']:+.3f} dB/°C, interval "
              f"[{n['db_per_c_lo']:+.3f}, {n['db_per_c_hi']:+.3f}] contains zero)")

        r = e.loc[("sensor02", "gatewayC", "rssi")]
        assert not r["primary"] and abs(r["db_per_c"] - BETA) < 0.02, r
        assert tiers[("sensor04", "gatewayB")] == "mid_range", tiers
        print(f"  saturated    ok   (RSSI {r['db_per_c']:+.3f} dB/°C reported, but scored on PDR)")

        gw = res["gateways"].set_index("gateway")
        a, b = gw.loc["gatewayA"], gw.loc["gatewayB"]
        assert abs(a["noise_db"]) < 0.3, a["noise_db"]                    # real loss: floor still
        assert abs(b["noise_db"] - BETA * sw) < 0.3, b["noise_db"]        # artefact: floor moves
        assert abs(b["snr_db"]) < 0.3 and abs(b["rssi_db"] - BETA * sw) < 0.3, b
        print(f"  receiver     ok   (real loss on A: noise floor {a['noise_db']:+.2f} dB; "
              f"gain drift on B: RSSI {b['rssi_db']:+.2f}, SNR {b['snr_db']:+.2f}, "
              f"noise floor {b['noise_db']:+.2f} dB)")

        p = res["pdr"].set_index(["sensor", "gateway"]).loc[("sensor05", "gatewayC")]
        assert p["primary"] and abs(p["pp_per_c"] - PDR_BETA * 100) < 0.2, p["pp_per_c"]
        print(f"  censored     ok   (scored on PDR: {p['pp_per_c']:+.2f} pp/°C vs "
              f"{PDR_BETA * 100:+.0f} planted)")

        d = res["diurnal"].set_index(["sensor", "gateway", "metric"]).loc[("sensor02", "gatewayA", "snr")]
        assert d["slope_db_per_c"] < 0 and abs(d["slope_db_per_c"] - BETA) < 0.06, d["slope_db_per_c"]
        assert d["weekday_minus_weekend_db"] < -0.5, d["weekday_minus_weekend_db"]
        print(f"  diurnal      ok   (slope {d['slope_db_per_c']:+.3f} dB/°C; people effect seen as "
              f"weekday−weekend {d['weekday_minus_weekend_db']:+.2f} dB)")

        st = res["steps"]
        hit = st[(st["sensor"] == "sensor02") & (st["gateway"] == "gatewayA")]
        assert any(abs((pd.Timestamp(x) - STEP_DATE.tz_localize(None)).days) <= 3
                   for x in hit["date"]), hit
        print(f"  step         ok   (found {hit['date'].tolist()} for the planted 2024-07-15)")

        ne = res["natural"].set_index(["pair", "gateway"]).loc[("02/03", "gatewayA")]
        assert ne["difference_hi"] < 0 and abs(ne["difference"] - BETA * sw) < 0.4, ne
        print(f"  natural exp  ok   (outdoor−indoor {ne['difference']:+.2f} dB vs "
              f"{BETA * sw:+.2f} planted, interval excludes zero)")

        assert v["links"] == 3 and abs(v["pooled_swing_db"] - BETA * sw / 3) < 0.3, v
        assert v["pooled_lo"] < v["pooled_swing_db"] < v["pooled_hi"], v
        print(f"  verdict      ok   ({v['label']}: pooled {v['pooled_swing_db']:+.2f} dB "
              f"[{v['pooled_lo']:+.2f}, {v['pooled_hi']:+.2f}] over {v['links']} links)")

        md = w2.report(res, v)
        assert "Verdict" in md and "Natural experiment" in md and "Where the effect lives" in md
        print("  report       ok")
        print("\nall checks passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run()
