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

PySpark requires a local JVM (Java 11+). TLC trip records are not committed (see `.gitignore`). Run the downloads scripts below first:


## Repo Structure
```
/README.md      -- you are here

/data/processed     -- derived .csv and .parquet files used by the notebooks for modelling and analysis

/models     -- fitted model weights (.pkl/.joblib), committed so demand_model.ipynb can be read without refitting

/notebooks      -- containing all analysis/ modelling notebooks

/scripts        -- containing all scripts necessary for dataframe aggregation/ downloading supplementary datasets

/plots      -- raw .png/ .html files containing plots used within the report

/report     -- report.tex source, report.pdf render references.bib

```

## Quick Start

### Downloading datasets

Below are examples of using `downloads.py`, graders are recommended to use the first example to download only external datasets. 

```bash
# For downloading external datasets only
python3 scripts/downloads.py --dataset external

# To download all datasets 
python3 scripts/downloads.py --dataset all

# Or for individual datasets...
python3 scripts/downloads.py --dataset yellow green events

# Choices: yellow, green, fhvhv, monthly-trip-counts, weather, events, taxi-zones
```

### Pre-processing

Pre-processing is handled by various python files, processed data is sent to `data/processed`, later utilised by the notebooks for analysis/ modelling. Run these after the downloads above -- `process_weather.py` needs `intraday_15min_by_date_mode_2025.csv`, so it must come after the aggregation scripts.

```bash
# Distributions/outliers (general_stats.ipynb) + daily/intraday/event/airport
# summaries (timeseries_and_summaries.ipynb) -- one pass over the raw trips
python3 scripts/aggregate_distribution_stats.py

# Per-zone pickup counts by mode (market_share_maps.ipynb)
python3 scripts/aggregate_zone_counts.py

# Hourly + daily weather joined with trip volume (timeseries_and_summaries.ipynb)
# -- run after aggregate_distribution_stats.py above (needs its intraday output)
python3 scripts/process_weather.py

# To build the design matrix (demand_model.ipynb)
python3 scripts/generate_design_matrix.py
```

### Notebooks

Once the above has run, the four notebooks are independent of each other and can be run in any order:

- `general_stats.ipynb` -- outlier analysis (boxplots, adopted bounds)
- `market_share_maps.ipynb` -- zone-level choropleths (taxi/rideshare share, demand)
- `timeseries_and_summaries.ipynb` -- holidays, weather effects, typical-week profile, long-run 2009-2025 context
- `demand_model.ipynb` -- the Poisson/NB demand model, GBM benchmark, Uncaptured Demand results

`demand_model.ipynb` caches each slow model fit to `models/*.pkl`/`.joblib` on first run and just reloads them after -- delete the relevant file (or the whole folder) to force a refit, e.g. if you've regenerated the design matrix.

