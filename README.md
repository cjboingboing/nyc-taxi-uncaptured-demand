# MAST30034 Project 1 - Uncaptured Demand

Quantifies **Uncaptured Demand**: the trip volume yellow/green taxis are losing to rideshare (Uber/Lyft) in each NYC taxi zone, for taxi fleet operators and dispatch planners. Full write-up: [`report/report.pdf`](report/report.pdf)
(source: [`report/report.tex`](report/report.tex)).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
# venv/Scripts/activate on Windows

pip install -r requirements.txt
```

PySpark requires a local JVM (Java 11+). TLC trip records are not commited (see `.gitignore`). Run the downlaods scripts below first:


## Repo Structure
```
/README.md      -- you are here

/data/processed     -- derived .csv and .parquet files used by the notebooks for modelling and analysis

/models     -- .gitignored (recreated by demand_model.ipynb to save model weights once training is complete)

/notebooks      -- containing all analysis/ modelling notebooks

/scripts        -- containing all scripts necessary for dataframe aggregation/ downloading supplementary datasets

/plots      -- raw .png/ .html files containing plots used within the report

/report     -- report.tex source, report.pdf render references.bib

```

## Quick Start

Downloading datasets

```bash
# For downloading external datasets only
python3 scripts/downloads.py --dataset external

# To download all datasets 
python3 scripts/downloads.py --dataset all

# Or for individual datasets...
python3 scripts/downloads.py --dataset yellow green events

# Choices: yellow, green, fhvhv, monthly-trip-counts, weather, events
```