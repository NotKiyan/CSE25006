"""Static facts about the UVA LoRaWAN deployment, plus pipeline constants.

Every deployment fact here is copied from the dataset README
(Nikseresht, Leal Sobral, Goodall, Campbell 2025; doi:10.18130/V3/RFTICK).
Distances are COMPUTED from the README coordinates, not copied, so they can
be re-derived if a coordinate is ever corrected.

Nothing in this file is a measurement. It is metadata only.
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------
# Regions.  Three different ones are in play in this project; keep them apart.
#   UVA_REGION        - the band the dataset itself was recorded in.
#   SIM_REGION        - the band ns-3 experiments run in.
#   DEPLOY_REGION     - the band the real deployment targets.
# UVA is used to learn how climate moves a link (dB), never to source
# region-specific SF / TP / airtime rules.  Those come from SIM/DEPLOY specs.
# --------------------------------------------------------------------------
UVA_REGION = "US915"      # inferred from the Cisco IXM-LPWA-900 gateways; verified in 02
SIM_REGION = "EU868"
DEPLOY_REGION = "IN865"

# UVA campus, Charlottesville VA.  Handles EST/EDT automatically.
LOCAL_TZ = "America/New_York"

# Deployment window stated in the README.
COVERAGE_START = "2024-02-09"
COVERAGE_END = "2025-06-11"

# --------------------------------------------------------------------------
# Sensors (README "Sensor and Gateway Metadata")
# --------------------------------------------------------------------------
SENSORS = {
    "sensor01": dict(model="Linovision S500CO2", lat=38.0328110, lon=-78.5095524,
                     indoor=True,  floor=3, placement="office, 3rd floor of a 3-storey building"),
    "sensor02": dict(model="Linovision S500TH",  lat=38.0321764, lon=-78.5107493,
                     indoor=False, floor=0, placement="near the entrance door of an office building"),
    "sensor03": dict(model="Linovision S500TH",  lat=38.0321568, lon=-78.5108207,
                     indoor=True,  floor=2, placement="lab, 2nd floor of a 2-storey building"),
    "sensor04": dict(model="Linovision S500TH",  lat=38.0327080, lon=-78.5097230,
                     indoor=False, floor=0, placement="near the entrance door of an office building"),
    "sensor05": dict(model="Linovision S500TH",  lat=38.0333228, lon=-78.5111476,
                     indoor=False, floor=0, placement="corridor between two office buildings"),
    "sensor06": dict(model="Linovision S500TH",  lat=38.0332850, lon=-78.5111832,
                     indoor=True,  floor=3, placement="office, 3rd floor of a 3-storey building"),
    "sensor07": dict(model="Linovision S500TH",  lat=38.0328110, lon=-78.5095524,
                     indoor=True,  floor=3, placement="office, 3rd floor of a 3-storey building"),
    "sensor08": dict(model="Linovision S500TH",  lat=38.0347279, lon=-78.5135257,
                     indoor=False, floor=0, placement="near the entrance door of an office building"),
    "sensor09": dict(model="Linovision S500TH",  lat=38.0346453, lon=-78.5134932,
                     indoor=True,  floor=1, placement="classroom, single-storey office building"),
    "sensor10": dict(model="Linovision S500TH",  lat=38.0318946, lon=-78.5102145,
                     indoor=True,  floor=2, placement="office, 2nd floor of a 2-storey building"),
}

GATEWAYS = {
    "gatewayA": dict(model="Cisco IXM-LPWA-900-16-K9", lat=38.0317195, lon=-78.5108640,
                     indoor=False, placement="roof of a 5-storey building"),
    "gatewayB": dict(model="Cisco IXM-LPWA-900-16-K9", lat=38.0365606, lon=-78.5225695,
                     indoor=False, placement="roof of a single-storey building"),
    "gatewayC": dict(model="Cisco IXM-LPWA-900-16-K9", lat=38.0322023, lon=-78.5106216,
                     indoor=True,  placement="2nd floor of a 2-storey building"),
}

# --------------------------------------------------------------------------
# The natural experiment (Week 2).
#
# CONTRAST pairs: two sensors metres apart, one indoors and one outdoors.
# The outdoor radio swings with ambient temperature; the indoor one sits at
# room temperature behind the same wall.  If only the outdoor node's dSNR
# tracks temperature, the effect lives in the node hardware.  If both track
# it, the effect lives in the path or the gateway.
#
# REPLICATE pair: two sensors at identical coordinates, both indoors, same
# placement.  Their difference is device-to-device variability with climate
# held constant -- it is the noise floor the contrasts must beat.
# --------------------------------------------------------------------------
CONTRAST_PAIRS = [
    ("sensor02", "sensor03"),   # outdoor / indoor,  ~7 m apart
    ("sensor05", "sensor06"),   # outdoor / indoor,  ~5 m apart
    ("sensor08", "sensor09"),   # outdoor / indoor, ~10 m apart
]
REPLICATE_PAIRS = [
    ("sensor01", "sensor07"),   # both indoor, identical coordinates
]

# --------------------------------------------------------------------------
# Column canonicalisation.  README names on the left, short names used
# everywhere downstream on the right.
# --------------------------------------------------------------------------
COLUMN_MAP = {
    "timestamp": "ts",
    "sensor alias": "sensor",
    "gateway alias": "gateway",
    "rssi (dbm)": "rssi",
    "snr (db)": "snr",
    "spreading factor (-)": "sf",
    "bandwidth (hz)": "bw",
    "frequency (hz)": "freq",
    "airtime (s)": "airtime",
    "counter (-)": "counter",
    "# receiving gateways (-)": "n_hear_reported",
}

# Keyword fallback, used only when an exact name misses.  Order matters:
# the first keyword that appears in the lowered column name wins.
# Specific keywords sit before generic ones: "airtime" must win over "time",
# and "receiving gateways" must win over "gateway", or the fallback would
# map airtime onto the timestamp and the n_hear column onto the gateway.
COLUMN_KEYWORDS = [
    ("airtime", "airtime"),
    ("receiving", "n_hear_reported"),
    ("timestamp", "ts"), ("time", "ts"),
    ("sensor", "sensor"), ("device", "sensor"),
    ("gateway", "gateway"),
    ("rssi", "rssi"), ("snr", "snr"),
    ("spreading", "sf"), ("bandwidth", "bw"), ("frequency", "freq"),
    ("counter", "counter"), ("fcnt", "counter"),
]

REQUIRED_COLUMNS = ["ts", "sensor", "gateway", "rssi", "snr", "sf", "counter"]

# Bumped whenever the pipeline's output changes meaning.  The data card
# refuses to profile a build made by an older version.
PIPELINE_VERSION = 4

# --------------------------------------------------------------------------
# Pipeline constants.  One place, so a sweep later changes one file.
# --------------------------------------------------------------------------
# Receptions of the SAME (sensor, counter) closer than this are one
# transmission.  Must exceed any gateway backhaul lag or clock skew, and stay
# far below the time it takes a device to reboot and reach the same counter
# value again.  10 minutes satisfies both by a wide margin.
UPLINK_GROUP_TOLERANCE_S = 600

# PDR is reconstructed from frame-counter gaps inside this bin.
PDR_BIN = "1h"

# A PDR bin with fewer expected uplinks than this is too noisy to use.
PDR_MIN_EXPECTED = 3

# A counter that jumps forward by more than this inside one epoch is treated
# as a lost stretch too long to be a plausible gap -- the bin is dropped
# rather than scored as near-zero PDR.
PDR_MAX_GAP = 500

# A weather outage longer than this is reported as a named gap.
WEATHER_GAP_MIN_HOURS = 24

# US915 uplink plan, used only to VERIFY the region, never to set policy.
US915_FREQ_MIN_HZ = 902_000_000
US915_FREQ_MAX_HZ = 928_000_000
EU868_FREQ_MIN_HZ = 863_000_000
EU868_FREQ_MAX_HZ = 870_000_000
IN865_FREQ_MIN_HZ = 865_000_000
IN865_FREQ_MAX_HZ = 867_000_000


# --------------------------------------------------------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    radius = 6_371_000.0
    p1, l1, p2, l2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin((l2 - l1) / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def link_distances() -> dict[tuple[str, str], float]:
    """Metres from every sensor to every gateway, keyed (sensor, gateway)."""
    out = {}
    for s, sm in SENSORS.items():
        for g, gm in GATEWAYS.items():
            out[(s, g)] = haversine_m(sm["lat"], sm["lon"], gm["lat"], gm["lon"])
    return out


def gateway_separations() -> dict[tuple[str, str], float]:
    """Metres between each pair of gateways."""
    names = list(GATEWAYS)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[(a, b)] = haversine_m(
                GATEWAYS[a]["lat"], GATEWAYS[a]["lon"],
                GATEWAYS[b]["lat"], GATEWAYS[b]["lon"],
            )
    return out


def pair_separation(s1: str, s2: str) -> float:
    """Metres between two sensors."""
    a, b = SENSORS[s1], SENSORS[s2]
    return haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])


# ==========================================================================
# Week 2 -- the climate gate.  Every threshold the analysis uses is here.
# ==========================================================================
# Lowest SNR a LoRa demodulator decodes at 125 kHz (Semtech datasheet).
SF_FLOOR_DB = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}

ANALYSIS_SF = 7                    # 9 of 10 sensors never left SF7
EXCLUDE_SENSORS = {"sensor10"}     # 17-20 days of data: no seasons to compare
MIN_RX_PER_LINK = 1000

# Link tiers, decided from the data, not by hand.
SAT_WINDOW_DB = 0.5        # "at the ceiling" = within this of the link's p99.9
SAT_SHARE_SATURATED = 0.20 # above this share at the ceiling: SNR cannot move
SAT_SHARE_NEAR = 0.05      # above this: SNR clips some of the time
MARGINAL_HEADROOM_DB = 2.5 # p05 SNR closer than this to the floor: censored
FLOOR_PDR = 0.10           # below this PDR the link only logs its good moments

# Day and night, by the sun rather than the clock, with twilight dropped.
DAY_SOLAR_ELEV = 10.0
NIGHT_SOLAR_ELEV = -6.0
MIN_HOURS_PER_SIDE = 10    # per week, for the diurnal contrast

# Fixed-effects estimate.
MIN_WEEKS = 8
N_BOOT = 300               # week-block bootstrap resamples
BOOT_SEED = 7

# Tail drop: hot vs mild afternoons, same season, same hours.
HOT_C = 30.0
MILD_C = (18.0, 25.0)
SUMMER_MONTHS = (5, 6, 7, 8, 9)
AFTERNOON_HOURS = (12, 18)         # inclusive, local time
MIN_TAIL_SAMPLES = 200

# Step changes (antenna knocked, firmware, new obstruction).
STEP_SHORT_DAYS = 7
STEP_LONG_DAYS = 28
STEP_SHORT_DB = 3.0
STEP_LONG_DB = 2.0

# The gate, in dB of SNR moved by a hot afternoon against a typical hour.
# Yardstick: one SF step buys ~2.5 dB, one TP step 3 dB.
GATE_LEADS_DB = 1.5
GATE_MODULATES_DB = 0.5

# CASM (Lavdas et al. 2026), free-space, SF7, high-vs-low temperature split.
CASM_PRIOR_SF7_DB = -0.75
