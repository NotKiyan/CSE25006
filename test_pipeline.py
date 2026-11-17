"""Unit tests for the UVA pipeline.

READ THIS FIRST.  The fixtures below are NOT data and never become a result.
They are a handful of rows whose correct answer is known by construction --
"18 of 20 counters survive, gateway B hears only the even ones" -- so that
the dedup, epoch and PDR arithmetic is checked before it is pointed at the
real archive.  No number from this file appears in any table, plot or paper.
Delete it and nothing about the analysis changes.

    python test_pipeline.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import uva_config as cfg
import uva_pipeline as up


def make_receptions() -> pd.DataFrame:
    """A deliberately awkward fixture with answers known in advance.

    sensor02, epoch 0: counters 0..19 at one-minute spacing, except that
      counters 5 and 11 never reached any gateway.  So 18 uplinks survive
      out of an expected span of 20.
        gatewayA hears every surviving uplink            -> 18/20 = 0.90
        gatewayB hears only even counters                -> 10/20 = 0.50
        gatewayC hears nothing                           ->  0/20 = 0.00
    sensor02, epoch 1: the device reboots two hours later and sends
      counters 0..5, all heard by gatewayA alone         ->  6/6  = 1.00
    Plus three rows that must be thrown away: an exact duplicate, an SNR of
    999 dB, and an alias the README does not define.
    """
    t0 = pd.Timestamp("2024-06-01 12:00", tz="UTC")
    rows = []

    for c in range(20):
        if c in (5, 11):                      # lost at every gateway
            continue
        ts = t0 + pd.Timedelta(minutes=c)
        rows.append(("sensor02", "gatewayA", ts, c, -95.0, 3.5, 7))
        if c % 2 == 0:
            # Gateway B logs its copy 90 s late -- after gateway A has already
            # heard the NEXT counter.  The first version split these into two
            # uplinks; this is the regression check for that bug.
            rows.append(("sensor02", "gatewayB", ts + pd.Timedelta(seconds=90),
                         c, -105.0, -2.0, 7))

    t1 = t0 + pd.Timedelta(hours=2)
    for c in range(6):
        rows.append(("sensor02", "gatewayA", t1 + pd.Timedelta(minutes=c), c, -97.0, 2.0, 8))

    # sensor04: a loss that sits exactly on an hour boundary.  Counters 0..9
    # land in the 13:00 hour, 10 and 11 are lost, 12..19 land in the 14:00
    # hour.  Scoring 14:00 from its own first counter would call it 8/8 =
    # 1.00 and hide the loss; chaining from 13:00's last counter gives 8/10.
    t2 = pd.Timestamp("2024-06-01 13:00", tz="UTC")
    for c in range(20):
        if c in (10, 11):
            continue
        ts = t2 + pd.Timedelta(minutes=5 * c) if c < 10 else \
            pd.Timestamp("2024-06-01 14:00", tz="UTC") + pd.Timedelta(minutes=5 * (c - 12))
        rows.append(("sensor04", "gatewayA", ts, c, -96.0, 3.0, 7))

    df = pd.DataFrame(rows, columns=["Sensor Alias", "Gateway Alias", "Timestamp",
                                     "Counter (-)", "RSSI (dBm)", "SNR (dB)",
                                     "Spreading Factor (-)"])
    # Rows that cleaning must remove.
    bad = pd.DataFrame([
        ("sensor02", "gatewayA", t0, 0, -95.0, 3.5, 7),        # exact duplicate
        ("sensor02", "gatewayA", t0 + pd.Timedelta(hours=5), 99, -95.0, 999.0, 7),
        ("sensor99", "gatewayA", t0 + pd.Timedelta(hours=6), 1, -95.0, 3.0, 7),
    ], columns=df.columns)
    return pd.concat([df, bad], ignore_index=True)


def run() -> None:
    raw = up.canonicalise(make_receptions())
    raw["ts"] = pd.to_datetime(raw["ts"], utc=True)
    raw["counter"] = raw["counter"].astype("Int64")
    raw["sf"] = raw["sf"].astype("Int64")

    rec, log = up.clean(raw)
    assert log.get("dropped_exact_duplicate_rows") == 1, log
    assert log.get("dropped_snr_out_of_range") == 1, log
    assert log.get("dropped_unknown_sensor") == 1, log
    print(f"  cleaning   ok   ({log['rows_in']} -> {log['rows_out']} rows)")

    rec = up.assign_uplinks(rec)
    assert rec["uplink_id"].nunique() == 42, rec["uplink_id"].nunique()   # 18 + 6 + 18
    epochs = sorted(rec["epoch"].unique())
    assert epochs == [0, 1], epochs
    assert rec.loc[rec["epoch"] == 1, "uplink_id"].nunique() == 6
    print(f"  uplinks    ok   (42 unique, epochs {[int(e) for e in epochs]})")

    rec = up.enrich(rec)
    assert rec["distance_m"].notna().all()
    # Midday in June at 38 N: the sun must be well up; midnight well down.
    noon = up.solar_elevation(38.03, -78.51, pd.Series([pd.Timestamp("2024-06-21 17:00", tz="UTC")]))
    midnight = up.solar_elevation(38.03, -78.51, pd.Series([pd.Timestamp("2024-06-21 05:00", tz="UTC")]))
    assert 60 < noon[0] < 80, noon
    assert midnight[0] < -10, midnight
    print(f"  enrich     ok   (solar elevation {noon[0]:.1f} deg at local noon, "
          f"{midnight[0]:.1f} deg at local midnight)")

    packets = up.build_packets(rec)
    assert len(packets) == 42
    first_epoch = packets[(packets["sensor"] == "sensor02") & (packets["epoch"] == 0)]
    assert (first_epoch["n_hear"] == 2).sum() == 10, "even counters reach two gateways"
    assert (first_epoch["n_hear"] == 1).sum() == 8, "odd counters reach one"
    assert (first_epoch["gateway_best"] == "gatewayA").all(), "A has the higher SNR"
    assert np.isclose(first_epoch[first_epoch["n_hear"] == 2]["snr_spread"].iloc[0], 5.5)
    print("  packets    ok   (n_hear 2 on 10 uplinks, 1 on 8; spread 5.5 dB)")

    pdr = up.reconstruct_pdr(packets, rec)
    s02 = pdr[pdr["sensor"] == "sensor02"]
    e0 = s02[s02["epoch"] == 0].set_index("gateway")["pdr"]
    assert np.isclose(e0["gatewayA"], 0.90), e0
    assert np.isclose(e0["gatewayB"], 0.50), e0
    assert np.isclose(e0["gatewayC"], 0.00), e0
    assert np.isclose(s02[s02["epoch"] == 0]["pdr_any"].iloc[0], 0.90)
    e1 = s02[s02["epoch"] == 1].set_index("gateway")["pdr"]
    assert np.isclose(e1["gatewayA"], 1.00), e1
    print("  pdr        ok   (A 0.90, B 0.50, C 0.00; after reboot A 1.00)")

    s04 = pdr[(pdr["sensor"] == "sensor04") & (pdr["gateway"] == "gatewayA")].sort_values("bin")
    assert np.allclose(s04["pdr"].to_numpy(), [1.0, 0.8]), s04[["bin", "expected", "pdr"]]
    print("  pdr edge   ok   (loss on the hour boundary: 13:00 1.00, 14:00 0.80, not 1.00)")

    # Geometry: the two gateways on the same block are ~58 m apart and both
    # about 1.15 km from gateway B.  These come straight from the README.
    sep = cfg.gateway_separations()
    assert 50 < sep[("gatewayA", "gatewayC")] < 70, sep
    assert 1100 < sep[("gatewayA", "gatewayB")] < 1250, sep
    for a, b in cfg.CONTRAST_PAIRS:
        assert cfg.pair_separation(a, b) < 20, (a, b)
        assert cfg.SENSORS[a]["indoor"] != cfg.SENSORS[b]["indoor"], (a, b)
    for a, b in cfg.REPLICATE_PAIRS:
        assert cfg.pair_separation(a, b) < 1.0, (a, b)
        assert cfg.SENSORS[a]["indoor"] == cfg.SENSORS[b]["indoor"], (a, b)
    print("  geometry   ok   (A-C 58 m, A-B 1.16 km, contrast pairs indoor/outdoor)")

    print("\nall checks passed")


if __name__ == "__main__":
    run()
