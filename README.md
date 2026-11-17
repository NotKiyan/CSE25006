# Environment-Aware ADR for LoRaWAN: Data and Simulation Tools

This repository holds the data and simulation tools of an M.Tech project on adaptive data rate (ADR) for LoRaWAN under varying climate. It contains:

- the pipeline that turns four public LoRaWAN datasets into clean, comparable tables;
- the analysis that measures how temperature affects SNR, RSSI and the gateway noise floor;
- a fit of the real links' signal variation over time, used to set up the simulator;
- ns-3 components for a fading channel, a climate-driven loss model and two published ADR baselines;
- scripts that check the ns-3 LoRaWAN module against published results.

The ADR controller proposed in the project is not part of this repository.

Authors: Charan, Darshan Balu and Dr. Vishnu S, Department of Computer Science and Engineering, Amrita School of Computing, Amrita Vishwa Vidyapeetham, Coimbatore, India.

## Main findings from these tools

- On the University of Virginia (UVA) campus deployment, ambient heat did not change SNR (within about 0.25 dB per 10 °C). The rooftop gateway's RSSI and noise floor fell together by about 4.5 dB per 10 °C. That pattern points to the gateway's receiver, not the radio path, so RSSI is unsafe as an ADR input where gateways sit in the sun.
- Heat at the end device does cost SNR: about 1.8 dB per 10 °C, pooled over two greenhouses and a polyhouse.
- The UVA links vary on two timescales: a slow component with a correlation time of several days, and a fast within-hour component of about 5.2 dB.

## Repository layout

| Path | Purpose |
| --- | --- |
| `uva_config.py` | Dataset constants: sensors, gateways, coordinates, frequency bands, cleaning thresholds |
| `uva_pipeline.py` | Builds clean tables from the raw UVA dataset: UTC time, deduplicated uplinks, lost uplinks restored from frame counters, hourly weather |
| `uva_data_card.py` | Profiles the built tables and writes `DATA_CARD.md` |
| `inspect_datasets.py` | Prints the schema of every dataset file before a loader is written for it |
| `datasets_common.py` | Converts each dataset into one standard reception table, so one analysis runs on all of them |
| `uva_week2.py` | Temperature effect on the UVA links: week and hour fixed effects, week-block bootstrap intervals |
| `cross_climate.py` | The same estimator run on every dataset in the standard format |
| `uva_channel_fit.py` | Temporal statistics of the UVA links (slow and fast variation, autocorrelation) for the simulator |
| `ns3_aloha_check.py` | Compares the ns-3 LoRaWAN module with pure ALOHA throughput |
| `ns3_finnegan_check.py` | Compares ns-3's ADR with the results of Finnegan et al. (IEEE IoT Journal, 2020) |
| `ns3_caadr_check.py` | Independent re-implementation of CA-ADR and Cluster-ADR, used to check the ns-3 versions |
| `test_pipeline.py`, `test_week2.py`, `test_cross.py` | Known-answer tests for the pipeline and the climate analysis |
| `ns3/` | C++ components for the ns-3 lorawan module (see below) |

## Requirements

- Python 3.11 or newer
- The packages in `requirements.txt`
- For the simulation part: ns-3.48 with the lorawan module

```bash
python -m venv .venv
source .venv/bin/activate        # fish: source .venv/bin/activate.fish
pip install -r requirements.txt
```

## Datasets

The datasets are not included in this repository. Download them from their original sources:

| Dataset | Source |
| --- | --- |
| UVA campus, US915 | F. Nikseresht, V. A. Leal Sobral, J. L. Goodall and B. Campbell, University of Virginia Dataverse, doi:10.18130/V3/RFTICK |
| Antwerp greenhouses | R. K. Singh, M. H. Rahmani, M. Weyn and R. Berkvens, Zenodo, doi:10.5281/zenodo.5793685 |
| Siegen indoor office | N. M. Obiri and K. Van Laerhoven, Zenodo, doi:10.5281/zenodo.15349731 |
| Medellín urban | M. González-Palacio et al., *Data*, vol. 8, no. 1, 2023, doi:10.3390/data8010004 |

Place the unzipped UVA dataset in one folder and the other datasets under `../datasets`.

## How to run

1. Look at the files before loading them (optional):

   ```bash
   python inspect_datasets.py --root ../datasets --uva-build ./build > datasets_report.txt
   ```

2. Build the clean UVA tables:

   ```bash
   python uva_pipeline.py --inspect --root /path/to/uva_dataset
   python uva_pipeline.py --build   --root /path/to/uva_dataset --out ./build
   python uva_data_card.py --build ./build --out DATA_CARD.md
   ```

3. Measure the temperature effect on the UVA links:

   ```bash
   python uva_week2.py --build ./build --out ./week2
   ```

4. Convert every dataset to the standard format, then run the same estimator on all of them:

   ```bash
   python datasets_common.py --help                 # lists the datasets it can load
   python datasets_common.py uva ./build --out ./standard
   python cross_climate.py --standard ./standard --out ./cross
   ```

5. Fit the temporal statistics of the UVA links:

   ```bash
   python uva_channel_fit.py --build ./build --out ./channel
   ```

Each step writes a Markdown report into its output folder.

## Tests

The tests use small fixtures with known answers. They check the code, and none of their data appears in any result.

```bash
python test_pipeline.py
python test_week2.py
python test_cross.py
python uva_channel_fit.py --selftest
python ns3_caadr_check.py --selftest
python ns3_finnegan_check.py --selftest
```

## ns-3 components

The `ns3/` folder holds C++ components written for the ns-3 lorawan module:

| File | Purpose |
| --- | --- |
| `correlated-fading-loss-model.{h,cc}` | Per-link signal variation on three timescales (slow, short, per packet), fitted to the UVA links |
| `climate-trace.{h,cc}`, `climate-loss-model.{h,cc}` | Replays a real temperature and humidity trace into the channel as extra path loss |
| `collision-aware-adr-component.{h,cc}` | CA-ADR (Marini et al., 2021) and Cluster-ADR (Gorrela and Choudhury, 2025) as baselines |

To use them, copy the files into `src/lorawan/model/` of an ns-3.48 tree, add them to `src/lorawan/CMakeLists.txt` under `source_files` and `header_files`, then reconfigure and build ns-3. The components are licensed under GPL-2.0-only, like ns-3.

### Simulator checks

- `ns3_aloha_check.py` reads lines of the form `nDevices seed sent received`, produced by running the ns-3 `aloha-throughput` example over a range of device counts, and compares them with the pure ALOHA curve.
- `ns3_finnegan_check.py fit` fits the reference path loss to the published data-rate distribution. `ns3_finnegan_check.py check <dir>` compares ns-3 runs in `<dir>` with it.
- `ns3_caadr_check.py <run.txt> --clusters K --period T` checks the ns-3 CA-ADR (`K = 1`) or Cluster-ADR (`K = 3`) assignment against the independent implementation.

## Citing the datasets

If you use these tools, cite the dataset papers:

- F. Nikseresht, V. A. Leal Sobral, J. L. Goodall and B. Campbell, "Dataset: Long-term LoRaWAN communication metadata from an urban deployment," in *Proc. ACM BuildSys*, 2025, pp. 412–417.
- R. K. Singh, M. H. Rahmani, M. Weyn and R. Berkvens, "Joint communication and sensing: A proof of concept and datasets for greenhouse monitoring using LoRaWAN," *Sensors*, vol. 22, no. 4, p. 1326, 2022.
- N. M. Obiri and K. Van Laerhoven, "A comprehensive data description for LoRaWAN path loss measurements in an indoor office setting: Effects of environmental factors," *IEEE Access*, vol. 13, pp. 83148–83170, 2025.
- M. González-Palacio, D. Tobón-Vallejo, L. M. Sepúlveda-Cano, S. Rúa, G. Pau and L. B. Le, "LoRaWAN path loss measurements in an urban scenario including environmental effects," *Data*, vol. 8, no. 1, p. 4, 2023.
