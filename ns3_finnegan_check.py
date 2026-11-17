"""Week 5, check 3 -- does ns-3's ADR reproduce Finnegan et al. (IEEE IoT J 2020)?

Published target, Table VI (N = 1000, 5 km disk, one gateway, 600 s, margin 5 dB):
share of devices settling at each data rate, for all devices starting at DR0 and
for all starting at DR5.  Values read from the paper's table:

    end DR      0      1      2      3     4     5
    start DR0  36.0   23.4   14.2   8.7   5.5  12.2
    start DR5   8.3   32.5   20.9  12.6   8.3  17.4

The paper gives log-distance path loss with exponent 3 and no shadowing, but not
its link budget.  With no shadowing, the DR0-start column depends on ONE number,
the ADR's SNR at 1 m (call it C): a device at distance d sees SNR = C - 30 log10 d
on every packet, and the network-side ADR moves it from DR0 straight to
min(5, floor((SNR + 20 - 5) / 3)).  Devices are uniform on the disk, so each
share is a ring area.  `fit` estimates C from that column and converts it to the
--refLossDb the ns-3 scenario needs.  The DR5 column, which runs through the
device-side backoff and its interplay with the server, is then out of sample.

Three things are checked from the runs:

  exact   DR0 start: every device's final DR (and TP) must equal the value the
          ADR arithmetic gives for its own distance -- an implementation check,
          independent of the paper
  fit     DR0 column vs Table VI (in-sample: only C was fitted)
  test    DR5 column vs Table VI (out of sample: nothing re-tuned)

Packet delivery (Table VII) is printed for reference only: Finnegan's module
decodes chunk by chunk from AWGN bit-error curves, ours uses a fixed SNR
threshold, so delivery is not expected to match.

    python ns3_finnegan_check.py fit
    python ns3_finnegan_check.py check <dir> [refLossDb] [--offset dB]
    python ns3_finnegan_check.py --selftest
"""

from __future__ import annotations

import glob
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

TABLE_VI = {0: np.array([36.0, 23.4, 14.2, 8.7, 5.5, 12.2]),
            5: np.array([8.3, 32.5, 20.9, 12.6, 8.3, 17.4])}
TABLE_VII_PDR = {0: 90.79, 5: 75.85}
R_M = 5000.0
EXPONENT = 3.0
MARGIN_DB = 5.0
THR = [-20.0, -17.5, -15.0, -12.5, -10.0, -7.5]          # DR0..DR5, ns-3 AdrComponent
TP_MAX, TP_MIN = 14, 2
NOISE_DB = 174 - 10 * math.log10(125_000) - 6            # AdrComponent: SNR = Prx + this
TOL_POINTS = 3.0                                         # per-DR share tolerance (pp)


def shares_disk(c: float) -> np.ndarray:
    """DR0 start, no shadowing, uniform disk: share ending at each DR."""
    ge = [min(1.0, (10 ** ((c - s) / (10 * EXPONENT)) / R_M) ** 2)
          for s in (-12.0, -9.0, -6.0, -3.0, 0.0)]       # SNR needed for DR >= 1..5
    cum = [1.0] + ge + [0.0]
    return np.array([cum[i] - cum[i + 1] for i in range(6)]) * 100


def fit_c() -> float:
    r = minimize_scalar(lambda c: ((shares_disk(c) - TABLE_VI[0]) ** 2).sum(),
                        bounds=(80, 110), method="bounded")
    return float(r.x)


def ref_loss(c: float) -> float:
    return TP_MAX + NOISE_DB - c


GW_SENS_DBM = [-130.0, -132.5, -135.0, -137.5, -140.0, -142.5]   # ns-3 gateway, SF7..SF12


def heard_from_dr5(d: float, ref_loss_db: float, offset_db: float) -> int:
    """DR5 start, threshold reception: the device backs off one DR at a time
    until the gateway hears it (rx >= sensitivity + offset)."""
    rx = TP_MAX - ref_loss_db - 10 * EXPONENT * math.log10(max(d, 1.0))
    dr = 5
    while dr > 0 and rx < GW_SENS_DBM[(12 - dr) - 7] + offset_db:
        dr -= 1
    return dr


def predict(d: float, ref_loss_db: float, start_dr: int = 0) -> tuple[int, int]:
    """Final (DR, TP) for a DR0-start device at distance d: the ns-3 ADR step
    rule applied every evaluation until it stops changing anything."""
    dr, tp = start_dr, TP_MAX
    for _ in range(20):
        snr = tp - ref_loss_db - 10 * EXPONENT * math.log10(max(d, 1.0)) + NOISE_DB
        steps = math.floor((snr - THR[dr] - MARGIN_DB) / 3)
        ndr, ntp = dr, tp
        while steps > 0 and ndr < 5:
            ndr, steps = ndr + 1, steps - 1
        while steps > 0 and ntp > TP_MIN:
            ntp, steps = ntp - 2, steps - 1
        while steps < 0 and ntp < TP_MAX:
            ntp, steps = ntp + 2, steps + 1
        if (ndr, ntp) == (dr, tp):
            break
        dr, tp = ndr, ntp
    return dr, tp


def read_run(path: str) -> tuple[np.ndarray, tuple[float, float] | None]:
    rows, pdr = [], None
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == "PDR":
                pdr = (float(p[1]), float(p[2]))
            elif len(p) == 4 and p[0].isdigit():
                rows.append([float(x) for x in p])
    return np.array(rows), pdr


def check(folder: str, ref_loss_db: float, offset_db: float = 0.0) -> bool:
    ok = True
    for start in (0, 5):
        files = sorted(glob.glob(os.path.join(folder, f"dr{start}_s*.txt")))
        if not files:
            print(f"start DR{start}: no runs in {folder} (not part of this set)")
            continue
        per_seed, sent, recv, exact_dr, exact_tp, n_dev = [], 0.0, 0.0, 0, 0, 0
        xtab, never = np.zeros((6, 6), int), 0
        exp5, match5 = [], 0
        for fp in files:
            a, pdr = read_run(fp)
            if a.size == 0:
                print(f"  {fp}: no device lines")
                ok = False
                continue
            per_seed.append(np.bincount(a[:, 2].astype(int), minlength=6)[:6] / len(a) * 100)
            if pdr:
                sent, recv = sent + pdr[0], recv + pdr[1]
            if start == 0:
                pred = [predict(d, ref_loss_db) for d in a[:, 1]]
                for p, r, t in zip(pred, a[:, 2], a[:, 3]):
                    xtab[p[0], int(r)] += 1
                    if p[0] != r and int(r) == 0 and t == TP_MAX:
                        never += 1
                exact_dr += sum(int(p[0] == r) for p, r in zip(pred, a[:, 2]))
                exact_tp += sum(int(abs(p[1] - t) <= 2) for p, t in zip(pred, a[:, 3]))
                n_dev += len(a)
            else:
                pred5 = [predict(d, ref_loss_db, heard_from_dr5(d, ref_loss_db, offset_db))[0] for d in a[:, 1]]
                exp5.append(np.bincount(pred5, minlength=6)[:6] / len(a) * 100)
                match5 += sum(int(p == r) for p, r in zip(pred5, a[:, 2]))
                n_dev += len(a)
        if not per_seed:
            continue
        s = np.array(per_seed)
        m, sd = s.mean(0), s.std(0, ddof=1) if len(s) > 1 else np.zeros(6)
        diff = m - TABLE_VI[start]
        role = "in-sample (C fitted on this column)" if start == 0 else "OUT OF SAMPLE"
        print(f"\nstart DR{start} -- {len(s)} seeds -- {role}")
        print(f"  {'end DR':>6} {'ns-3':>7} {'±sd':>5} {'paper':>7} {'diff':>6}")
        for k in range(6):
            print(f"  {k:>6} {m[k]:>6.1f}% {sd[k]:>5.1f} {TABLE_VI[start][k]:>6.1f}% {diff[k]:>+6.1f}")
        worst = float(np.abs(diff).max())
        passed = worst <= TOL_POINTS
        ok &= passed
        print(f"  worst gap {worst:.1f} points (tolerance {TOL_POINTS:g}) -> {'PASS' if passed else 'FAIL'}")
        if start == 5 and exp5:
            e = np.mean(exp5, 0)
            print(f"  threshold-reception expectation (gateway sensitivity offset {offset_db:+g} dB, "
                  f"device backs off until heard):")
            print("  " + "  ".join(f"DR{k} {e[k]:.1f}%" for k in range(6)))
            print(f"  devices ending exactly there: {match5 / n_dev:.1%}")
        if sent:
            print(f"  delivery, last 50 periods: {100 * recv / sent:.1f}%  "
                  f"(paper {TABLE_VII_PDR[start]:.1f}%, reference only -- different PHY model)")
        if start == 0 and n_dev:
            share_dr, share_tp = exact_dr / n_dev, exact_tp / n_dev
            print(f"  exact check: final DR equals the ADR arithmetic for {share_dr:.1%} of devices, "
                  f"TP within one 2 dB step for {share_tp:.1%}")
            ok &= share_dr >= 0.98
            print("  predicted DR (rows) vs final DR in ns-3 (columns), devices summed over seeds:")
            print("        " + "".join(f"{k:>7}" for k in range(6)))
            for k in range(6):
                print(f"  {k:>6}" + "".join(f"{v:>7}" for v in xtab[k]))
            miss = n_dev - exact_dr
            if miss:
                print(f"  of the {miss} mismatched devices, {never} ({never / miss:.0%}) still sit at DR0 / "
                      f"{TP_MAX} dBm: the starting setting, i.e. no LinkADRReq ever reached them")
    return ok


def selftest() -> None:
    # Fixtures, not data; nothing here becomes a result.
    c = fit_c()
    assert 94 < c < 98, c
    assert np.abs(shares_disk(c) - TABLE_VI[0]).max() < 2.5
    rl = ref_loss(c)
    assert predict(100.0, rl) == (5, TP_MIN)                 # next to the gateway
    assert predict(4999.0, rl)[0] == 0                       # disk edge stays at DR0
    # a DR0-start fixture drawn from the arithmetic itself must pass the exact check
    rng = np.random.default_rng(1)
    d = R_M * np.sqrt(rng.random(1000))
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_finnegan_fixture")
    os.makedirs(tmp, exist_ok=True)
    with open(os.path.join(tmp, "dr0_s1.txt"), "w") as f:
        for i, x in enumerate(d):
            dr, tp = predict(x, rl)
            f.write(f"{i} {x} {dr} {tp}\n")
        f.write("PDR 1000 900\n")
    a, pdr = read_run(os.path.join(tmp, "dr0_s1.txt"))
    assert len(a) == 1000 and pdr == (1000.0, 900.0)
    for fp in glob.glob(os.path.join(tmp, "*")):
        os.remove(fp)
    os.rmdir(tmp)
    print(f"selftest ok (C = {c:.2f} dB -> refLossDb {rl:.2f})")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        selftest()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        c = fit_c()
        print(f"C (ADR SNR at 1 m) = {c:.2f} dB  ->  --refLossDb={ref_loss(c):.2f}")
        print(f"SNR at the 5 km edge: {c - 10 * EXPONENT * math.log10(R_M):.1f} dB "
              f"(SF12 floor -20: every device reachable)")
        p = shares_disk(c)
        print("  end DR   paper   disk model")
        for k in range(6):
            print(f"  {k:>6} {TABLE_VI[0][k]:>6.1f}% {p[k]:>8.1f}%")
        return
    if len(sys.argv) > 2 and sys.argv[1] == "check":
        args = sys.argv[3:]
        off = 0.0
        if "--offset" in args:
            i = args.index("--offset")
            off = float(args[i + 1])
            args = args[:i] + args[i + 2:]
        rl = float(args[0]) if args else ref_loss(fit_c())
        ok = check(sys.argv[2], rl, off)
        print(f"\noverall: {'PASS' if ok else 'FAIL'}")
        return
    print(__doc__)


if __name__ == "__main__":
    main()
