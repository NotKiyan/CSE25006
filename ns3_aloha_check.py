"""Week 5 step 1 -- does the ns-3 lorawan module reproduce pure ALOHA?

aloha-throughput: nDevices devices, each sends ONE packet at a uniformly
random time in a window of W = simulationTime seconds, all at SF7, 125 kHz,
with the 'aloha' interference matrix (any overlap destroys both).  Pure ALOHA
then predicts, for each packet, success = no other start within +-T of it:

    P(success) = (1 - 2T/W)^(N-1)  ~  exp(-2 (N-1) T / W)

T is the time on air of the transmitted frame.  The app payload is 50 B;
what the MAC adds on top is checked here rather than assumed: the script
fits both "no header" and "+13 B LoRaWAN overhead" and reports which one
the simulator matches.

Input: lines "N seed sent received" (see the shell loop in the message).

    python ns3_aloha_check.py aloha_sweep.txt
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd


def toa_s(phy_bytes: int, sf: int = 7, bw: int = 125_000) -> float:
    tsym = 2 ** sf / bw
    de = 1 if (sf >= 11 and bw == 125_000) else 0
    n = 8 + max(math.ceil((8 * phy_bytes - 4 * sf + 28 + 16) / (4 * (sf - 2 * de))) * 5, 0)
    return (8 + 4.25) * tsym + n * tsym


def p_aloha(n: int, t: float, w: float) -> float:
    return (1 - 2 * t / w) ** (n - 1)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        assert abs(toa_s(20) - 0.056576) < 1e-5            # Semtech calculator, 20 B at SF7
        assert abs(p_aloha(200, toa_s(63), 100) - 0.625) < 0.01
        print("selftest ok")
        return
    df = pd.read_csv(sys.argv[1], sep=r"\s+", names=["n", "seed", "sent", "received"])
    w = 100.0 if len(sys.argv) < 3 else float(sys.argv[2])
    g = df.groupby("n").agg(sent=("sent", "sum"), received=("received", "sum"), runs=("seed", "size"))
    g["sim"] = g["received"] / g["sent"]
    g["sim_lo"] = g["sim"] - 1.96 * np.sqrt(g["sim"] * (1 - g["sim"]) / g["sent"])
    g["sim_hi"] = g["sim"] + 1.96 * np.sqrt(g["sim"] * (1 - g["sim"]) / g["sent"])
    for label, phy in (("50 B, no header", 50), ("50 B + 13 B LoRaWAN", 63)):
        g[label] = [p_aloha(n, toa_s(phy), w) for n in g.index]
    print(f"window W = {w:g} s; ToA {toa_s(50) * 1000:.1f} ms (50 B) / {toa_s(63) * 1000:.1f} ms (63 B)\n")
    print(f"{'N':>6} {'runs':>4} {'simulated':>10} {'95% CI':>17} {'theory 50B':>11} {'theory 63B':>11}")
    for n, r in g.iterrows():
        print(f"{n:>6} {int(r['runs']):>4} {r['sim']:>10.3f}  [{r['sim_lo']:.3f}, {r['sim_hi']:.3f}] "
              f"{r['50 B, no header']:>11.3f} {r['50 B + 13 B LoRaWAN']:>11.3f}")
    for label in ("50 B, no header", "50 B + 13 B LoRaWAN"):
        inside = ((g[label] >= g["sim_lo"]) & (g[label] <= g["sim_hi"])).mean()
        err = (g["sim"] - g[label]).abs().max()
        print(f"\n{label}: theory inside the simulator's 95% CI at {inside:.0%} of N; worst gap {err:.3f}")
    print("\nA match (theory inside the CI at every N, gap < ~0.02) means the PHY, timing and collision "
          "model behave as published; the ADR baseline can then be trusted to sit on top of them.")


if __name__ == "__main__":
    main()
