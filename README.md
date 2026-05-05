# Interconnection Queue Analysis

Analysis of the U.S. grid interconnection queue using Berkeley Lab's [Queued Up](https://emp.lbl.gov/queues) 2025 dataset (data through end of 2024).

## What this does

Three analyses of ~10,300 generation projects waiting in interconnection queues across all U.S. RTOs:

1. **Queue concentration by Point of Interconnection (POI)** — which substations are the bottleneck nodes? Top-decile concentration POIs are flagged as risk clusters.
2. **Withdrawal probability model** — gradient boosting classifier predicting whether a queued project will be withdrawn vs. reach commercial operation. Historical baseline: ~86% of projects withdraw.
3. **Cluster-level upgrade cost analysis** — where are the most expensive network upgrade requirements concentrated, and which corridors are bottlenecks?

A Streamlit dashboard surfaces queue size by RTO over time, the highest-concentration POIs, and withdrawal probability distributions by resource type.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the Queued Up dataset (Excel file) from https://emp.lbl.gov/queues into `data/`.

## Run

```bash
# Run analyses
python -m src.load_data
python -m src.withdrawal_model
python -m src.concentration_analysis

# Launch dashboard
streamlit run app.py
```

## Project structure

```
interconnection-queue-analysis/
  data/                    # raw downloaded files (gitignored)
  notebooks/               # exploratory analysis
  src/
    load_data.py           # ingest, clean, normalize the Queued Up Excel
    withdrawal_model.py    # train gradient boosting classifier
    concentration_analysis.py  # POI concentration + upgrade cost clustering
    visualize.py           # shared plotting helpers
  app.py                   # streamlit dashboard
  README.md
```

## Data source

Berkeley Lab Electricity Markets and Policy Group, *Queued Up: 2025 Edition*. https://emp.lbl.gov/queues
