# Interconnection Queue Analysis

Two views of the U.S. grid interconnection queue stitched together: the historical record (Berkeley Lab's *Queued Up* 2025 dataset, through end of 2024) and a live tracker on top of PJM's planning API.

Built as a portfolio piece exploring the same data problem Tapestry (Alphabet) is solving for grid operators.

## What you see

A Streamlit dashboard organized as an executive narrative:

1. **Headline + hero KPIs** — active queue size, GW waiting, historical completion rate, median wait of active projects.
2. **Methodology panel (collapsed by default)** — source file, sheet, raw → canonical column mapping, label-derivation rules, POI sentinel filter.
3. **Live PJM tracker** — the most recent snapshot of PJM's queue (PJM is the largest U.S. RTO and Tapestry's first HyperQ deployment partner). KPIs for the active and in-flight cohorts, top 25 highest-risk active projects scored by the LBNL-trained model, and a placeholder for Cycle 1 (announced 2026-04-29; per-project data not public until validation closes 2026-07-27).
4. **Historical analysis** — POI concentration with top-decile risk clusters, withdrawal probability distribution by resource type, queue evolution by RTO since 2010.

## Pipeline

```
LBNL Queued Up workbook (.xlsx)
  └─ src/load_data.py         (sheet "03. Complete Queue Data", header row 1, Excel serial dates)
       └─ src/withdrawal_model.py     (gradient boosting classifier on resolved cohort)
       └─ src/concentration_analysis.py  (POI bottlenecks, sentinel bins filtered)

PJM planning API (services.pjm.com/PJMPlanningApi)
  └─ src/pjm_queue.py         (fetch + parquet snapshot per pull)
       └─ src/pjm_scoring.py  (bridge PJM columns → LBNL canonical, score with LBNL model)

→ app.py (Streamlit dashboard)
```

Snapshots accumulate as a public diff-able ledger of the PJM cohort's milestone progression in `data/pjm_snapshots/<date>.parquet` (~470 KB each).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the *Queued Up: 2025 Edition* workbook from https://emp.lbl.gov/queues into `data/`.

## Run

```bash
# Sanity-check the LBNL load and train + evaluate the model
python -m src.load_data
python -m src.withdrawal_model
python -m src.concentration_analysis

# Pull a fresh PJM snapshot (idempotent — one parquet per calendar day)
python -m src.pjm_queue

# Score the live PJM active queue with the LBNL-trained model
python -m src.pjm_scoring

# Launch the dashboard
streamlit run app.py
```

## Project structure

```
interconnection-queue-analysis/
  data/
    *.xlsx                      # LBNL Queued Up workbook (gitignored)
    pjm_snapshots/<date>.parquet # PJM live-queue snapshots (committed)
  src/
    load_data.py                # LBNL Excel → canonical schema
    withdrawal_model.py         # gradient boosting classifier
    concentration_analysis.py   # POI concentration + sentinel filtering
    pjm_queue.py                # PJM live-queue fetch + snapshot
    pjm_scoring.py              # bridge PJM → LBNL schema, score active queue
  app.py                        # Streamlit dashboard
  requirements.txt
  README.md
```

## Known caveat

The withdrawal model's absolute `P(withdraw)` clusters at 94–99% for renewables because LBNL's positive-class base rate is ~83%. Relative ranking across resource types and projects is meaningful; absolute calibration needs a held-out cohort. This is the next thing to fix.

## Data sources

- Berkeley Lab Electricity Markets and Policy Group, *Queued Up: 2025 Edition*. https://emp.lbl.gov/queues
- PJM Interconnection planning API: `services.pjm.com/PJMPlanningApi/api/Queue/ExportToXls` (endpoint and subscription key mined from PJM's public JS bundle; see [`gridstatus`](https://github.com/gridstatus/gridstatus) PJM module for reference).
