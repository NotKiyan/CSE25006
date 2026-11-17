"""Known-answer tests for datasets_common.py and cross_climate.py.

READ THIS FIRST.  As with the other test files, the fixtures are NOT data
and never become a result.  They are built in Medellin's exact file format
(day-first dates, local Bogota time, same columns, zipped) and with signals
composed the way they physically combine (RSSI = signal + noise in power),
so that the loader and the "where does the effect live" logic are checked
against answers known in advance:

  medellin  EN1, EN2: real path loss, signal -0.10 dB/°C -> SNR and RSSI fall,
                      noise flat.  EN3, EN4: no effect.
  uva-like  gateway A: gain drift -0.30 dB/°C -> RSSI and noise fall, SNR flat.
            gateway C: no effect.

    python test_cross.py       (about two minutes: it runs the bootstrap)
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import cross_climate as cc
import datasets_common as dc

PATH_BETA, GAIN_BETA = -0.10, -0.30


def weather(local_times: pd.DatetimeIndex, rng, base=21.0, amp=6.0) -> np.ndarray:
    hod = local_times.hour.to_numpy() + local_times.minute.to_numpy() / 60
    days = (local_times - local_times[0]).days.to_numpy()
    front = pd.Series(rng.standard_normal(days.max() + 1)).rolling(3, min_periods=1).mean().to_numpy() * 3
    wobble = pd.Series(rng.standard_normal(len(local_times))).rolling(12, min_periods=1).mean().to_numpy() * 1.5
    return base + amp * np.sin((hod - 8) / 24 * 2 * np.pi) + front[days] + wobble


def combine(signal_dbm, noise_dbm):
    """Received power and SNR from signal and noise, as a gateway reports them."""
    rssi = 10 * np.log10(10 ** (signal_dbm / 10) + 10 ** (noise_dbm / 10))
    return np.round(rssi), np.round((signal_dbm - noise_dbm) * 4) / 4


def medellin_fixture(path: Path, rng) -> None:
    local = pd.date_range("2021-11-04 00:00", "2022-03-20 23:55", freq="5min")
    temp = weather(local, rng)
    rows = []
    for node, beta, base in (("EN1", PATH_BETA, -95), ("EN2", PATH_BETA, -97),
                             ("EN3", 0.0, -96), ("EN4", 0.0, -98)):
        sig = base + beta * (temp - 21) + 1.0 * rng.standard_normal(len(local))
        noi = -104 + 0.8 * rng.standard_normal(len(local))
        rssi, snr = combine(sig, noi)
        sf = 7 + (np.arange(len(local)) % 4)
        rows.append(pd.DataFrame({
            "index": 0, "timestamp": [f"{t.day}/{t.month}/{t.year} {t.hour}:{t.minute:02d}" for t in local],
            "device_id": node, "distance": 2140, "sf": sf, "frequency": 904100000,
            "temperature": np.round(temp + 0.3 * rng.standard_normal(len(local)), 1),
            "rh": np.clip(80 - 2 * (temp - 21) + 8 * rng.standard_normal(len(local)), 5, 100), "rssi": rssi, "snr": snr,
            "pn": rssi - 10 * np.log10(1 + 10 ** (snr / 10))}))
    csv = pd.concat(rows).to_csv(index=False)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("LoRaWAN Path Loss Measurement Campaign No Outliers V2.csv", csv)


def uva_like_fixture(rng) -> pd.DataFrame:
    utc = pd.date_range("2024-03-01", "2024-08-31 23:50", freq="10min", tz="UTC")
    temp = weather(utc.tz_convert("America/New_York").tz_localize(None), rng, base=18, amp=7)
    rows = []
    for gw, gain in (("A", GAIN_BETA), ("C", 0.0)):
        for node in ("s01", "s04", "s08"):
            sig = -100 + 1.0 * rng.standard_normal(len(utc))
            noi = -112 + 0.8 * rng.standard_normal(len(utc))
            drift = gain * (temp - 18)          # the gateway's own reading shifts
            rssi, snr = combine(sig + drift, noi + drift)
            rows.append(pd.DataFrame({"node": node, "gateway": gw, "sf": 7, "ts": utc,
                                      "snr": snr, "rssi": rssi, "temp_c": temp, "rh_pct": 60.0}))
    # The trap: a strong link whose SNR reading tops out at +12 dB, with a
    # REAL signal loss in heat.  It mimics the gateway pattern exactly.
    for node in ("s02", "s05"):
        sig = -60 - 0.30 * (temp - 18) + rng.standard_normal(len(utc))
        rssi, snr = combine(sig, -112 + 0.8 * rng.standard_normal(len(utc)))
        snr = np.minimum(snr, 12.0) - np.abs(0.6 * rng.standard_normal(len(utc)))
        rows.append(pd.DataFrame({"node": node, "gateway": "D", "sf": 7, "ts": utc,
                                  "snr": snr, "rssi": rssi, "temp_c": temp, "rh_pct": 60.0}))
    return dc.finish(pd.concat(rows), dc.SITES["uva"])


def greenhouse_fixture(folder: Path, rng) -> None:
    """Real sheet layout: per-sensor AF sheets, pivot columns, START-END, a glitch."""
    folder.mkdir(parents=True, exist_ok=True)
    local = pd.date_range("2020-04-10 13:00", "2020-06-10 23:50", freq="10min")
    temp = weather(local, rng, base=22, amp=7)
    def sheet(offset):
        t = np.round(temp + 0.2 * rng.standard_normal(len(local)), 2)
        t[4000] = -45.0                                  # the real file has this glitch
        sig = -105 - 0.10 * (t - 22)
        rssi, snr = combine(sig + rng.standard_normal(len(local)), -110 + 0.5 * rng.standard_normal(len(local)))
        return pd.DataFrame({"Date": local.normalize(), "Time": local.strftime("%H:%M:%S"),
                             "Onboard Temperature": t, "Onboard Humidity": 70.0 + offset,
                             "RSSI": rssi, "SNR": snr})
    with pd.ExcelWriter(folder / "Greenhouse-1.xlsx") as x:
        sheet(0).to_excel(x, sheet_name="AF16", index=False)
    with pd.ExcelWriter(folder / "Greenhouse-2.xlsx") as x:
        g2 = sheet(1)
        g2["Unnamed: 6"] = np.nan
        g2["Row Labels"] = "x"
        g2["Average of RSSI"] = -100.0
        g2.to_excel(x, sheet_name="AF49", index=False)
        pd.DataFrame({"AF49": [1, 2]}).to_excel(x, sheet_name="TEMP", index=False)
        pd.DataFrame({"Sensor": ["AF49"], "Start Date": [pd.Timestamp("2020-05-01")],
                      "End Date": [pd.Timestamp("2020-05-31")]}).to_excel(x, sheet_name="START-END", index=False)


def siegen_fixture(path: Path, rng) -> None:
    """Office: people raise CO2, warm the room AND attenuate the signal.

    Temperature itself does nothing.  Occupied days are random, so people
    vary inside each week and hour -- exactly what fools a model without
    an occupancy control.
    """
    utc = pd.date_range("2024-09-26", "2025-03-31", freq="5min", tz="UTC")
    local = utc.tz_convert("Europe/Berlin")
    day = (local.normalize() - local.normalize()[0]).days.to_numpy()
    occupied_day = rng.random(day.max() + 1) < 0.6
    office_hours = np.asarray((local.hour >= 9) & (local.hour < 17) & (local.dayofweek < 5))
    people = occupied_day[day] & office_hours
    temp = 20 + 2.0 * people + 1.2 * np.sin(day / 9) + 0.3 * rng.standard_normal(len(utc))
    co2 = 420 + 450 * people + 20 * rng.standard_normal(len(utc))
    rows = []
    for dev in ("ED0", "ED1"):
        sig = -75 - 3.0 * people + rng.standard_normal(len(utc))
        rssi, snr = combine(sig, -110 + 0.5 * rng.standard_normal(len(utc)))
        rows.append(pd.DataFrame({
            "time": [t.isoformat() for t in utc], "device_id": dev, "co2": co2,
            "humidity": 45 + 3 * rng.standard_normal(len(utc)), "temperature": temp,
            "rssi": rssi, "snr": snr, "SF": 9, "n_power": rssi - 10 * np.log10(1 + 10 ** (snr / 10))}))
    pd.concat(rows).to_csv(path, index=False)


def run() -> None:
    rng = np.random.default_rng(1)
    tmp = Path(tempfile.mkdtemp())
    try:
        z = tmp / "LoRaWAN Path Loss Measurement Campaign.zip"
        medellin_fixture(z, rng)
        med = dc.load_medellin(z)
        assert med["ts"].dt.tz is not None and str(med["ts"].dt.tz) == "UTC"
        first = med.sort_values("ts")["ts"].iloc[0]
        assert first == pd.Timestamp("2021-11-04 05:00", tz="UTC"), first   # 00:00 Bogota = 05:00 UTC
        assert med["unit"].nunique() == 16
        print(f"  loader       ok   (day-first dates, Bogota local time detected, 16 units)")

        # A clock shifted 5 h the wrong way must be refused, with clues,
        # and accepted once the shift is given.
        z2 = tmp / "shifted.zip"
        with zipfile.ZipFile(z) as zin, zipfile.ZipFile(z2, "w") as zout:
            name = zin.namelist()[0]
            d = pd.read_csv(zin.open(name))
            t = pd.to_datetime(d["timestamp"], dayfirst=True) - pd.Timedelta(hours=5)
            d["timestamp"] = [f"{x.day}/{x.month}/{x.year} {x.hour}:{x.minute:02d}" for x in t]
            zout.writestr(name, d.to_csv(index=False))
        try:
            dc.load_medellin(z2)
            raise AssertionError("a clock 5 h off should have been refused")
        except SystemExit:
            pass
        fixed = dc.load_medellin(z2, shift_hours=5)
        assert fixed.sort_values("ts")["ts"].iloc[0] == first, fixed["ts"].min()
        print("  clock        ok   (5 h error refused with clues; --shift-hours 5 restores it)")

        # Per-node constant offsets (the real Medellin pattern): accepted as-is.
        z3 = tmp / "per_node.zip"
        with zipfile.ZipFile(z) as zin, zipfile.ZipFile(z3, "w") as zout:
            d = pd.read_csv(zin.open(name))
            t = pd.to_datetime(d["timestamp"], dayfirst=True)
            t = t + pd.to_timedelta(d["device_id"].map({"EN1": 7, "EN2": 13, "EN3": -6, "EN4": -2}), unit="h")
            d["timestamp"] = [f"{x.day}/{x.month}/{x.year} {x.hour}:{x.minute:02d}" for x in t]
            zout.writestr(name, d.to_csv(index=False))
        asis = dc.load_medellin(z3, clock="as-is")
        assert len(asis) == len(med)
        print("  as-is clock  ok   (constant per-node offsets accepted after the drift check)")

        gh = dc.load_greenhouse(_gh(tmp, rng))
        assert gh["ts"].min() >= pd.Timestamp("2020-04-17 06:38", tz="Europe/Brussels"), gh["ts"].min()
        g2 = gh[gh["node"] == "AF49"]
        assert g2["ts"].min() >= pd.Timestamp("2020-05-01", tz="Europe/Brussels")
        assert g2["ts"].max() < pd.Timestamp("2020-06-01 01:00", tz="Europe/Brussels")
        assert gh["temp_c"].min() > -10 and set(gh["unit"]) == {"AF16→GH1@SF?", "AF49→GH2@SF?"}
        assert gh["temp_c"].isna().sum() >= 1, "the -45 °C reading inside the window must be blanked"
        print("  greenhouse   ok   (bench period, START-END window and -45 °C glitch removed)")

        sg_path = tmp / "2.aggregated_measurements_data.csv"
        siegen_fixture(sg_path, rng)
        sg = dc.load_siegen(sg_path)
        assert str(sg["ts"].dt.tz) == "UTC" and sg["co2_ppm"].notna().all()
        us, ps = cc.analyse_dataset(sg, rng)
        r = ps.iloc[0]
        assert r["snr_lo"] < 0 < r["snr_hi"], (r["snr_per10"], r["snr_lo"], r["snr_hi"])
        no_co2 = cc.analyse_dataset(sg.assign(co2_ppm=np.nan), rng)[1].iloc[0]
        assert no_co2["snr_hi"] < 0, no_co2[["snr_per10", "snr_lo", "snr_hi"]]
        print(f"  siegen       ok   (with CO2: SNR {r['snr_per10']:+.2f} dB/10°C, interval holds 0; "
              f"without CO2 the people effect poses as heat: {no_co2['snr_per10']:+.2f})")

        u, p = cc.analyse_dataset(med, rng)
        en1 = u[u["node"] == "EN1"]
        # Each unit carries a quarter of the packets, so judge it by whether its
        # 95% interval holds the planted value, and the average by closeness.
        assert ((en1["snr_lo"] < PATH_BETA * 10) & (PATH_BETA * 10 < en1["snr_hi"])).sum() >= 3, \
            en1[["snr_per10", "snr_lo", "snr_hi"]]
        assert abs(en1["snr_per10"].mean() - PATH_BETA * 10) < 0.15, en1["snr_per10"]
        assert abs(en1["noise_per10"].mean()) < 0.15, en1["noise_per10"]
        en3 = u[u["node"] == "EN3"]
        assert (en3["snr_lo"] < 0).all() and (en3["snr_hi"] > 0).all(), en3[["snr_lo", "snr_hi"]]
        pm = p.iloc[0]
        assert abs(pm["snr_per10"] - PATH_BETA * 10 / 2) < 0.2, pm["snr_per10"]
        print(f"  medellin     ok   (EN1 SNR {en1['snr_per10'].mean():+.2f} dB/10°C vs "
              f"{PATH_BETA * 10:+.1f} planted, noise {en1['noise_per10'].mean():+.2f}; "
              f"EN3 null; pooled {pm['snr_per10']:+.2f})")

        uv = uva_like_fixture(rng)
        u2, p2 = cc.analyse_dataset(uv, rng)
        a = p2.set_index("gateway").loc["A"]
        c = p2.set_index("gateway").loc["C"]
        assert a["reading"].startswith("gateway"), a["reading"]
        assert abs(a["rssi_per10"] - GAIN_BETA * 10) < 0.4 and abs(a["snr_per10"]) < 0.3, a
        assert c["reading"].startswith("no effect"), c["reading"]
        d = p2.set_index("gateway").loc["D"]
        assert (u2[u2["gateway"] == "D"]["tier"] == "compressed").all()
        assert not d["reading"].startswith("gateway"), d["reading"]
        print(f"  compressed   ok   (capped-SNR link with real loss flagged, not called a gateway: "
              f"'{d['reading']}')")
        print(f"  gateway      ok   (A: RSSI {a['rssi_per10']:+.2f}, noise {a['noise_per10']:+.2f}, "
              f"SNR {a['snr_per10']:+.2f} -> '{a['reading']}'; C: '{c['reading']}')")

        md = cc.report(pd.concat([u, u2]), pd.concat([p, p2]))
        assert "Climate effect across datasets" in md and "medellin" in md
        print("  report       ok")
        print("\nall checks passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _gh(tmp: Path, rng) -> Path:
    folder = tmp / "zenedoo greenhouse"
    greenhouse_fixture(folder, rng)
    return folder


if __name__ == "__main__":
    run()
