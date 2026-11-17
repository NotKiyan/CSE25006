"""Profile the built UVA tables and write DATA_CARD.md.

This answers the Week 1 questions in one pass and, more importantly, prints
the numbers that every later result has to be judged against:

  * the standard deviation of SNR on each link, and the error a
    predict-the-mean model would get.  The vineyard benchmark looked
    excellent at 0.193 dB MAE until that baseline turned out to be 0.295 dB.
    No model result is reported again without this column beside it.
  * whether the temperature range actually contains heat, rather than the
    winter-only range that made the vineyard set useless for extrapolation.
  * whether SF moves under ADR, which decides how much of any SNR change is
    the network reacting rather than the channel changing.

    python uva_data_card.py --build ./build --out DATA_CARD.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import uva_config as cfg
import uva_pipeline as up

MIN_DIURNAL_SAMPLES = 200      # per side, per link, at one SF
HOT_C = 30.0                   # "heat" for the extrapolation split, absolute
MIN_HOT_PER_LINK = 200         # receptions at or above HOT_C to call a link usable


def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    """Render a DataFrame as a GitHub pipe table."""
    def cell(v):
        if isinstance(v, float) and np.isfinite(v):
            return floatfmt.format(v)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        return str(v)
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


# ==========================================================================
def section_region(rec: pd.DataFrame) -> tuple[str, list[str]]:
    """Confirm the band the data was recorded in, from the data itself."""
    flags: list[str] = []
    lines = ["## Region, confirmed from the data",
             "",
             f"Simulations run in **{cfg.SIM_REGION}**, the real deployment targets "
             f"**{cfg.DEPLOY_REGION}**, and this dataset was recorded in a third band. "
             "UVA is used to learn how climate moves a link in dB, never to source "
             "SF, power or airtime rules.",
             ""]

    if "freq" in rec.columns and rec["freq"].notna().any():
        f = rec["freq"].dropna().astype("int64")
        lo, hi = int(f.min()), int(f.max())
        if cfg.US915_FREQ_MIN_HZ <= lo and hi <= cfg.US915_FREQ_MAX_HZ:
            verdict = "US915"
        elif cfg.EU868_FREQ_MIN_HZ <= lo and hi <= cfg.EU868_FREQ_MAX_HZ:
            verdict = "EU868"
        elif cfg.IN865_FREQ_MIN_HZ <= lo and hi <= cfg.IN865_FREQ_MAX_HZ:
            verdict = "IN865"
        else:
            verdict = "unrecognised"
        lines += [f"- Frequency range: {lo/1e6:.1f}–{hi/1e6:.1f} MHz over "
                  f"{f.nunique()} distinct channels → **{verdict}**"]
        if verdict != cfg.UVA_REGION:
            flags.append(f"Frequencies say {verdict}, but UVA_REGION is set to "
                         f"{cfg.UVA_REGION}. Fix uva_config.py before Week 2.")
    else:
        flags.append("No frequency column: the band could not be confirmed from the data.")

    sfs = sorted(rec["sf"].dropna().unique().tolist())
    lines.append(f"- Spreading factors present: {sfs}")
    if max(sfs) > 10:
        flags.append(f"SF above 10 present ({sfs}), which US915 125 kHz uplinks "
                     "do not use. Check the SF column's meaning.")
    if "bw" in rec.columns and rec["bw"].notna().any():
        bws = sorted(rec["bw"].dropna().unique().tolist())
        lines.append(f"- Bandwidths: {[int(b) for b in bws]} Hz")
    return "\n".join(lines), flags


def section_coverage(rec: pd.DataFrame, pdr: pd.DataFrame) -> tuple[str, list[str]]:
    """Per-link coverage, signal statistics and the baseline to beat."""
    flags: list[str] = []
    dist = cfg.link_distances()

    rows = []
    for (s, g), part in rec.groupby(["sensor", "gateway"], sort=True):
        snr = part["snr"].dropna()
        sigma = float(snr.std()) if len(snr) > 1 else np.nan
        link_pdr = pdr[(pdr["sensor"] == s) & (pdr["gateway"] == g)]["pdr"]
        rows.append({
            "link": f"{s[-2:]}→{g[-1]}",
            "m": round(dist[(s, g)]),
            "in/out": ("in" if cfg.SENSORS[s]["indoor"] else "out")
                      + "/" + ("in" if cfg.GATEWAYS[g]["indoor"] else "out"),
            "rx": len(part),
            "days": part["ts"].dt.normalize().nunique(),
            "SNR mean": float(snr.mean()) if len(snr) else np.nan,
            "SNR σ": sigma,
            # For Gaussian residuals a predict-the-mean model lands here.
            "mean-pred MAE": sigma * np.sqrt(2 / np.pi) if np.isfinite(sigma) else np.nan,
            "PDR": float(link_pdr.mean()) if len(link_pdr) else np.nan,
        })
    tab = pd.DataFrame(rows)

    thin = tab[tab["rx"] < 1000]
    if len(thin):
        flags.append(f"{len(thin)} of {len(tab)} links have under 1,000 receptions "
                     f"({', '.join(thin['link'])}). Treat them as anecdote, not evidence.")

    lines = ["## Links, and the baseline every model must beat",
             "",
             "`mean-pred MAE` is what a model that ignores every feature and "
             "predicts the link's mean SNR would score. A model that does not "
             "clearly beat this column has learned nothing, whatever its MAE "
             "looks like in isolation.",
             "",
             md_table(tab)]
    return "\n".join(lines), flags


def section_adr(rec: pd.DataFrame) -> tuple[str, list[str]]:
    """How much the network moved SF, which confounds every SNR comparison."""
    flags: list[str] = []
    rows = []
    for s, part in rec.sort_values("ts").groupby("sensor", sort=True):
        per_gw = part[part["gateway"] == part["gateway"].mode().iloc[0]]
        sf = per_gw["sf"].dropna()
        # Count transitions on plain floats: a nullable Int64 comparison
        # returns NA at the first element and silently swallows it.
        sfv = sf.to_numpy(dtype="float64", na_value=np.nan)
        changes = int(np.sum(sfv[1:] != sfv[:-1])) if len(sfv) > 1 else 0
        # year*12+month avoids to_period's tz-drop warning and is just as
        # good for counting distinct calendar months.
        ym = per_gw["ts"].dt.year * 12 + per_gw["ts"].dt.month
        months = max(ym.nunique(), 1)
        rows.append({
            "sensor": s,
            "SFs used": ",".join(str(int(v)) for v in sorted(sf.unique())),
            "SF changes": changes,
            "per month": changes / months,
            "modal SF": int(sf.mode().iloc[0]) if len(sf) else None,
        })
    tab = pd.DataFrame(rows)

    multi_sf = (tab["SFs used"].str.count(",") > 0).sum()
    if multi_sf:
        flags.append(f"{multi_sf} of {len(tab)} sensors used more than one SF. "
                     "Every climate comparison must be conditioned on SF, or the "
                     "effect measured is partly ADR reacting, not the channel moving.")

    lines = ["## ADR activity",
             "",
             "CASM states plainly that its temperature–SNR relationship was "
             "measured under a live ADR policy, so it mixes propagation with "
             "control. The same is true here. The table below sizes that problem.",
             "",
             md_table(tab)]
    return "\n".join(lines), flags


def section_multigateway(packets: pd.DataFrame) -> tuple[str, list[str]]:
    flags: list[str] = []
    dist = packets["n_hear"].value_counts().sort_index()
    rows = [{"gateways hearing it": int(k), "uplinks": int(v),
             "share": f"{v / len(packets):.1%}"} for k, v in dist.items()]

    lines = ["## Multi-gateway diversity",
             "",
             "CASM keeps only the highest-SNR reception of each packet and "
             "discards the rest, so none of this exists there. It is the raw "
             "material for the diversity-aware allocator in Week 4.",
             "",
             md_table(pd.DataFrame(rows))]

    if "n_hear_agrees" in packets.columns:
        agree = float(packets["n_hear_agrees"].mean())
        lines += ["", f"Reconstructed `n_hear` agrees with the dataset's own "
                      f"`# Receiving Gateways` column on **{agree:.1%}** of uplinks."]
        xt = pd.crosstab(packets["n_hear"], packets["n_hear_reported"])
        xt.index.name = None
        tab = xt.reset_index().rename(columns={"index": "reconstructed \\ reported"})
        tab.columns = [str(c) for c in tab.columns]
        lines += ["", "Rows = gateways heard in this data; columns = the dataset's own count.",
                  "", md_table(tab, floatfmt="{:.0f}")]
        above = int(sum(xt.loc[r, c] for r in xt.index for c in xt.columns if c > r))
        below = int(sum(xt.loc[r, c] for r in xt.index for c in xt.columns if c < r))
        max_reported = int(max(xt.columns))
        if agree < 0.95 and below == 0 and max_reported > len(cfg.GATEWAYS):
            # Settled, not open: the dataset reports more gateways than were
            # published, and our count never exceeds it, so dedup never
            # over-merges.  Record it as a fact about the deployment.
            lines += ["", f"**Finding:** the dataset reports up to {max_reported} "
                          f"gateways hearing one uplink, but only {len(cfg.GATEWAYS)} "
                          f"were published. Our count is never higher than the "
                          f"dataset's ({below} cases), so the deduplication never "
                          f"merges two packets. The {above:,} uplinks where it is lower "
                          "were heard by gateways not in the dataset. `n_hear` in "
                          "this card counts the three published gateways; "
                          "`n_hear_reported` counts the whole network. PDR here "
                          "is PDR over the three published gateways, a lower bound "
                          "on network PDR."]
        elif agree < 0.95:
            if above > 3 * max(below, 1):
                flags.append(f"n_hear agreement is {agree:.1%}, and the dataset's count "
                             f"exceeds ours on {above:,} uplinks vs the reverse on "
                             f"{below:,}. Most likely the network had gateways beyond "
                             "the three published. Use the reported column for "
                             "network-level diversity; ours for the three-gateway study.")
            else:
                flags.append(f"n_hear agreement is {agree:.1%} with no clear direction "
                             f"({above:,} above, {below:,} below). Send the crosstab "
                             "back before Week 2.")

    spread = packets.loc[packets["n_hear"] > 1, "snr_spread"].dropna()
    if len(spread):
        lines += ["", f"Where two or more gateways hear the same uplink, their SNRs "
                      f"differ by a median of **{spread.median():.1f} dB** "
                      f"(p90 {spread.quantile(0.9):.1f} dB). That spread is the "
                      "headroom a diversity-aware allocator can spend."]
    return "\n".join(lines), flags


def section_weather(rec: pd.DataFrame, weather: pd.DataFrame | None,
                    stats: dict) -> tuple[str, list[str]]:
    flags: list[str] = []
    lines = ["## Weather"]

    if weather is None:
        flags.append("No weather table was built. Week 2 cannot run without it.")
        return "\n".join(lines + ["", "Not available."]), flags

    tcol = up.weather_column(weather, "temp")
    hcol = up.weather_column(weather, "rh_pct", "humid")
    lines += ["", f"Columns: {', '.join(str(c) for c in weather.columns)}"]

    if tcol is None:
        flags.append("No temperature column found in the weather table.")
    else:
        # uva_pipeline.load_weather converts °F to °C automatically before
        # this table is ever written, so a high median here means that
        # conversion did not run on this build -- rebuild, don't re-flag.
        t = weather[tcol].dropna()
        if len(t) and t.median() > 50:
            flags.append(f"`{tcol}` has a median of {t.median():.0f}, still "
                         "°F-shaped. Rebuild with the current uva_pipeline.py "
                         "(the °F->°C conversion lives in load_weather).")
        lines += ["",
                  f"Temperature (`{tcol}`, °C): "
                  f"min {t.min():.1f}, p05 {t.quantile(.05):.1f}, "
                  f"median {t.median():.1f}, p95 {t.quantile(.95):.1f}, "
                  f"max {t.max():.1f}"]
        span_c = t.max() - t.min()
        lines.append(f"- Range spans about {span_c:.0f} °C.")
        if span_c < 25:
            flags.append(f"Temperature range is only ~{span_c:.0f} °C. The vineyard "
                         "set failed for exactly this reason. Check the coverage "
                         "before committing to a heat-extrapolation split.")

    if hcol is not None:
        h = weather[hcol].dropna()
        lines.append(f"- Humidity (`{hcol}`): median {h.median():.0f}, "
                     f"p05 {h.quantile(.05):.0f}, p95 {h.quantile(.95):.0f}")

    gaps = stats.get("weather_gaps", [])
    if gaps:
        lines += ["", "Outages longer than a day:", "",
                  md_table(pd.DataFrame(gaps)[["start", "end", "hours"]])]
        worst = max(gaps, key=lambda g: g["hours"])
        lines.append(f"\nThe longest is {worst['hours']/24:.0f} days "
                     f"({worst['start'][:10]} to {worst['end'][:10]}). Exclude these "
                     "windows from Week 2 rather than interpolating across them.")

    covered = stats.get("receptions_with_weather")
    if covered is not None:
        lines.append(f"\n**{covered:.1%}** of receptions fall in an hour that has weather.")
        if covered < 0.8:
            flags.append(f"Only {covered:.0%} of receptions have weather. "
                         "Check whether the gap overlaps the hottest months.")

    # Does real heat overlap the radio data?  An absolute threshold, counted
    # per link: a percentile cut would always hold 5% of the data by
    # construction and could never fail.
    if tcol is not None and tcol in rec.columns:
        hot = rec[rec[tcol] >= HOT_C]
        per_link = hot.groupby(["sensor", "gateway"]).size()
        good = per_link[per_link >= MIN_HOT_PER_LINK]
        lines.append(f"\nReceptions in hours at or above {HOT_C} °C: **{len(hot):,}**, "
                     f"over {hot['ts'].dt.floor('1h').nunique():,} distinct hours. "
                     f"Links with at least {MIN_HOT_PER_LINK} such receptions: "
                     f"**{len(good)}** of {rec.groupby(['sensor', 'gateway']).ngroups}.")
        if len(good) < 10:
            flags.append(f"Only {len(good)} links have {MIN_HOT_PER_LINK}+ receptions "
                         f"at or above {HOT_C} °C. The heat-extrapolation split may be "
                         "too thin to carry a result.")
    return "\n".join(lines), flags


def section_diurnal(rec: pd.DataFrame) -> tuple[str, list[str]]:
    """Can the primary (foliage-free) analysis actually run on these links?"""
    flags: list[str] = []
    if "is_daylight" not in rec.columns:
        return "", ["No solar elevation column; the diurnal analysis cannot be planned."]

    counts = (rec.groupby(["sensor", "gateway", "sf", "is_daylight"], sort=False)
                 .size().unstack(fill_value=0))
    counts.columns = ["night", "day"] if list(counts.columns) == [False, True] \
        else [str(c) for c in counts.columns]
    if "day" not in counts.columns or "night" not in counts.columns:
        return "", ["Receptions fall on only one side of sunrise; check the time zone."]

    usable = counts[(counts["day"] >= MIN_DIURNAL_SAMPLES)
                    & (counts["night"] >= MIN_DIURNAL_SAMPLES)]
    lines = ["## Diurnal analysis: is it feasible?",
             "",
             "Week 2 leads with day-versus-night inside the same week, because "
             "foliage is constant there and season is not. That only works where "
             "one link has enough samples on both sides of sunrise at the *same* SF.",
             "",
             f"- Link–SF combinations with at least {MIN_DIURNAL_SAMPLES} receptions "
             f"on both sides: **{len(usable)}** of {len(counts)}",
             f"- Sensors covered: **{usable.index.get_level_values('sensor').nunique()}** "
             f"of {len(cfg.SENSORS)}"]

    if len(usable) < 10:
        flags.append(f"Only {len(usable)} link–SF combinations support a diurnal "
                     "comparison. The primary analysis may have to fall back to "
                     "season, which reintroduces the foliage confound.")
    return "\n".join(lines), flags


def section_natural_experiment(rec: pd.DataFrame) -> str:
    rows = []
    for a, b in cfg.CONTRAST_PAIRS + cfg.REPLICATE_PAIRS:
        kind = "contrast" if (a, b) in cfg.CONTRAST_PAIRS else "replicate"
        na = int((rec["sensor"] == a).sum())
        nb = int((rec["sensor"] == b).sum())
        rows.append({
            "pair": f"{a[-2:]} / {b[-2:]}",
            "kind": kind,
            "apart (m)": round(cfg.pair_separation(a, b), 1),
            "placement": ("outdoor/indoor" if kind == "contrast" else "indoor/indoor"),
            f"rx (first)": na,
            f"rx (second)": nb,
            "usable": "yes" if min(na, nb) > 1000 else "thin",
        })
    return "\n".join([
        "## The natural experiment",
        "",
        "Three sensor pairs sit metres apart with one indoors and one outdoors. "
        "The outdoor radio swings with ambient temperature while its partner "
        "stays at room temperature behind the same wall. If only the outdoor "
        "node's ΔSNR tracks temperature, the effect lives in the node hardware; "
        "if both track it, it lives in the path or the gateway. Sensors 01 and "
        "07 sit at identical coordinates, both indoors — their difference is "
        "device-to-device variability with climate held constant, which is the "
        "noise floor the three contrasts have to beat.",
        "",
        md_table(pd.DataFrame(rows)),
    ])


# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="./build", help="folder written by uva_pipeline.py")
    ap.add_argument("--out", default="DATA_CARD.md")
    args = ap.parse_args()

    bd = Path(args.build)
    rec = pd.read_parquet(bd / "uva_receptions.parquet")
    packets = pd.read_parquet(bd / "uva_packets.parquet")
    pdr = pd.read_parquet(bd / "uva_pdr.parquet")
    wpath = bd / "uva_weather.parquet"
    weather = pd.read_parquet(wpath) if wpath.exists() else None
    stats = json.loads((bd / "build_stats.json").read_text())

    # A failed rebuild leaves the old tables in place, and profiling them
    # silently reports stale numbers.  Refuse instead.
    built = stats.get("pipeline_version")
    expected = getattr(cfg, "PIPELINE_VERSION", None)
    if expected is None:
        raise SystemExit(
            "uva_config.py is older than uva_data_card.py (it has no "
            "PIPELINE_VERSION). Replace uva_config.py and uva_pipeline.py with "
            "the latest versions, then rebuild.")
    if built != expected:
        raise SystemExit(
            f"{bd} was built by pipeline version {built}, but this is version "
            f"{expected}. The last --build probably failed; rerun it "
            "and check it ends with 'done ->'.")

    flags: list[str] = []
    parts: list[str] = []

    span = f"{rec['ts'].min():%Y-%m-%d} to {rec['ts'].max():%Y-%m-%d}"
    parts.append("\n".join([
        "# UVA LoRaWAN dataset — data card",
        "",
        f"Built {pd.Timestamp.now('UTC'):%Y-%m-%d} from `{stats.get('source')}`.",
        "",
        f"- Receptions after cleaning: **{len(rec):,}**",
        f"- Unique uplinks: **{len(packets):,}**",
        f"- Span: **{span}**",
        f"- Sensors: {rec['sensor'].nunique()} · Gateways: {rec['gateway'].nunique()} "
        f"· Links: {rec.groupby(['sensor','gateway']).ngroups}",
        "",
        "Citation: Nikseresht, Leal Sobral, Goodall and Campbell (2025), "
        "*Long-term LoRaWAN Communication Metadata from an Urban Deployment* (v1.0), "
        "University of Virginia Dataverse, doi:10.18130/V3/RFTICK (CC BY 4.0).",
        "",
        "## Cleaning",
        "",
        md_table(pd.DataFrame(
            [{"step": k, "rows": v} for k, v in stats.get("cleaning", {}).items()])),
    ]))

    for fn, arg in [(section_region, (rec,)),
                    (section_coverage, (rec, pdr)),
                    (section_adr, (rec,)),
                    (section_multigateway, (packets,)),
                    (section_weather, (rec, weather, stats)),
                    (section_diurnal, (rec,))]:
        text, f = fn(*arg)
        if text:
            parts.append(text)
        flags += f
    parts.append(section_natural_experiment(rec))

    if flags:
        parts.append("\n".join(["## Open issues", "",
                                "Resolve these before the Week 2 gate runs.", ""]
                               + [f"{i}. {f}" for i, f in enumerate(flags, 1)]))
    else:
        parts.append("## Open issues\n\nNone raised by the automatic checks.")

    Path(args.out).write_text("\n\n".join(parts) + "\n")
    print(f"wrote {args.out}")
    if flags:
        print(f"\n{len(flags)} open issue(s):")
        for i, f in enumerate(flags, 1):
            print(f"  {i}. {f}")


if __name__ == "__main__":
    main()
