"""Independent re-implementation of CA-ADR (Marini et al. 2021, Algorithm 3) and
Cluster-ADR (Gorrela & Choudhury 2025, Algorithm 1), used to check the ns-3
component collision-aware-adr-component.cc device by device.

With no shadowing every device's average received power is fixed by its
distance, so the allocation the ns-3 network server makes can be recomputed
here exactly from the scenario output ("id distance dr tp" lines).

    python ns3_caadr_check.py <run.txt> --clusters K --period T [--ref-loss 34.97]
    python ns3_caadr_check.py --selftest
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

PR_MIN = {7: -124.5, 8: -127.0, 9: -129.5, 10: -132.0, 11: -134.5, 12: -137.0}   # CA-ADR Table II
TP_MAX = 14.0
EXPONENT = 3.0


def toa_s(sf: int, phy_bytes: int, bw: int = 125_000, cr: int = 1, preamble: int = 8) -> float:
    """Semtech AN1200.13 time on air; explicit header, CRC on, LDRO for SF >= 11."""
    tsym = 2 ** sf / bw
    de = 1 if sf >= 11 else 0
    n = 8 + max(math.ceil((8 * phy_bytes - 4 * sf + 28 + 16) / (4 * (sf - 2 * de))) * (cr + 4), 0)
    return (preamble + 4.25) * tsym + n * tsym


def sf_min(pr: float) -> int:
    for sf in range(7, 13):
        if pr >= PR_MIN[sf]:
            return sf
    return 12


def allocate(pr: np.ndarray, period: float, phy_bytes: int, clusters: int = 1,
             per_cluster_cap: bool = True) -> np.ndarray:
    """Return the SF per device (input order kept)."""
    n = len(pr)
    order = np.arange(n)
    cluster = np.zeros(n, int)
    if clusters > 1:                                  # sort by path loss, K equal groups
        order = np.argsort(-pr, kind="stable")
        per = -(-n // clusters)
        cluster[order] = np.arange(n) // per
    groups = clusters if (clusters > 1 and per_cluster_cap) else 1
    smin = np.array([sf_min(p) for p in pr])
    pmac, res = 1.0, np.full(n, 12)
    for _ in range(100):
        count = np.zeros((groups, 6), int)
        assigned = 0
        for i in order:
            g = cluster[i] if groups > 1 else 0
            for sf in range(smin[i], 13):
                x = 1 - toa_s(sf, phy_bytes) / period
                nmax = 1 if x <= 0 else int(math.floor(0.5 * math.log(pmac) / math.log(x))) + 1
                if count[g, sf - 7] < nmax:
                    count[g, sf - 7] += 1
                    res[i] = sf
                    assigned += 1
                    break
        if assigned == n or pmac <= 0.01 + 1e-12:
            break
        pmac -= 0.01
    return res


def read(path: str) -> np.ndarray:
    rows = [l.split() for l in open(path) if l.strip() and l.split()[0].isdigit()]
    return np.array([[float(x) for x in r] for r in rows if len(r) == 4])


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        # Fixtures, not data; nothing here becomes a result.
        assert abs(toa_s(7, 20) - 0.056576) < 1e-5
        pr = np.array([-110.0] * 10)
        sf = allocate(pr, period=1.0, phy_bytes=21)
        assert len(set(sf)) > 1, "a tight cap must spread devices over SFs"
        assert (allocate(pr, period=1e9, phy_bytes=21) == 7).all(), "no cap -> lowest SF"
        print("selftest ok")
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--clusters", type=int, default=1)
    ap.add_argument("--period", type=float, default=600.0)
    ap.add_argument("--ref-loss", type=float, default=34.97)
    ap.add_argument("--bytes", type=int, nargs="+", default=[21, 22, 23, 24, 25])
    a = ap.parse_args()
    r = read(a.run)
    pr = TP_MAX - a.ref_loss - 10 * EXPONENT * np.log10(np.maximum(r[:, 1], 1.0))
    sf_ns3 = 12 - r[:, 2].astype(int)
    print(f"{len(r)} devices, K = {a.clusters}, T = {a.period:g} s")
    print(f"  ns-3 SF counts: {dict(zip(*np.unique(sf_ns3, return_counts=True)))}")
    best = None
    for b in a.bytes:             # the server sizes ToA from the last frame (FOpts vary)
        sf_py = allocate(pr, a.period, b, a.clusters)
        m = (sf_py == sf_ns3).mean()
        print(f"  PHY payload {b} B: python SF counts {dict(zip(*np.unique(sf_py, return_counts=True)))}, "
              f"device-level match {m:.1%}")
        best = max(best or 0, m)
    print(f"best match {best:.1%}")


if __name__ == "__main__":
    main()
