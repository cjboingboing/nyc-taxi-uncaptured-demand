"""
Single entry point for every dataset this project depends on that isn't
already sitting in this repo (all of them -- data/ is gitignored, see
.gitignore).

Graders reproducing the analysis already have the TLC trip parquet
(course-provided), so the option that actually matters here is `external`
-- 2025 weather, 2025 NYC permitted events, and the long-run monthly trip
counts, the small supplementary files the notebooks need that nothing else
provides. The TLC options (yellow/green/fhvhv) and `all` exist for
completeness / rebuilding this repo's data/ from scratch.

Usage:
    python downloads.py                          # interactive menu
    python downloads.py --dataset events weather  # non-interactive
    python downloads.py --dataset external         # weather + events + monthly-trip-counts
    python downloads.py --dataset all               # everything below

DATASET -> OUTPUT
    yellow                2025 yellow taxi trips (parquet)     -> data/tlc_data/{YYYY}-{MM}.parquet
    green                 2025 green taxi trips (parquet)      -> data/tlc_data/green_tripdata_{YYYY}-{MM}.parquet
    fhvhv                 2025 HVFHV rideshare trips (parquet) -> data/tlc_data/fhvhv_tripdata_{YYYY}-{MM}.parquet
    weather                2025 weather, Central Park daily
                           + LaGuardia hourly                  -> data/weather/central_park_weather_2025.csv
                                                                   data/weather/lga_hourly_weather_2025.csv
    events                 2025 NYC permitted events            -> data/export.csv
    monthly-trip-counts    long-run (2009-2025) monthly trip
                           counts by mode                       -> data/processed/monthly_trip_counts_2009_2025.csv

All filenames/schemas match exactly what scripts/tlc_trips.py,
scripts/aggregate_distribution_stats.py, scripts/generate_design_matrix.py,
and scripts/process_weather.py already expect -- nothing downstream needs to
change based on how a file got here.
"""

import argparse
import calendar
import io
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
TLC_DIR = ROOT / "data" / "tlc_data"
WEATHER_DIR = ROOT / "data" / "weather"
PROCESSED_DIR = ROOT / "data" / "processed"

YEAR = 2025
CHUNK_SIZE = 1024 * 1024
TLC_CDN = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def _fetch_bytes(url: str, headers: dict | None = None) -> tuple[bytes, dict]:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0", **(headers or {})})
    with urlopen(req, timeout=120) as resp:
        return resp.read(), dict(resp.headers)


def _download_month(cdn_name: str, out_path: Path) -> None:
    """Stream one month of TLC parquet from the public CDN, skipping if
    already present at the right size."""
    url = f"{TLC_CDN}/{cdn_name}"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        if out_path.exists() and out_path.stat().st_size == total:
            print(f"  [skip] {out_path.name} already downloaded ({total / 1e6:.1f} MB)")
            return
        print(f"  [download] {out_path.name} ({total / 1e6:.1f} MB)")
        tmp_path = out_path.with_suffix(".parquet.part")
        with open(tmp_path, "wb") as f:
            while chunk := resp.read(CHUNK_SIZE):
                f.write(chunk)
        tmp_path.rename(out_path)


def _download_trip_months(cdn_prefix: str, local_prefix: str, year: int) -> None:
    TLC_DIR.mkdir(parents=True, exist_ok=True)
    for month in range(1, 13):
        cdn_name = f"{cdn_prefix}{year}-{month:02d}.parquet"
        out_path = TLC_DIR / f"{local_prefix}{year}-{month:02d}.parquet"
        try:
            _download_month(cdn_name, out_path)
        except Exception as e:
            print(f"  [error] {year}-{month:02d}: {e}", file=sys.stderr)


def download_yellow(year: int = YEAR) -> None:
    print(f"Downloading {year} yellow taxi trips...")
    # Saved WITHOUT the "yellow_tripdata_" prefix -- tlc_trips.load_clean_trips
    # expects bare "{year}-{month}.parquet" for yellow specifically.
    _download_trip_months(cdn_prefix="yellow_tripdata_", local_prefix="", year=year)


def download_green(year: int = YEAR) -> None:
    print(f"Downloading {year} green taxi trips...")
    _download_trip_months(cdn_prefix="green_tripdata_", local_prefix="green_tripdata_", year=year)


def download_fhvhv(year: int = YEAR) -> None:
    print(f"Downloading {year} HVFHV (rideshare) trips...")
    _download_trip_months(cdn_prefix="fhvhv_tripdata_", local_prefix="fhvhv_tripdata_", year=year)


def _download_daily_weather(year: int) -> None:
    """Central Park (NOAA GHCN-D station USW00094728) daily summary."""
    out_path = WEATHER_DIR / f"central_park_weather_{year}.csv"
    station = "USW00094728"
    data_types = ["PRCP", "SNOW", "SNWD", "TMAX", "TMIN", "TAVG", "AWND"]
    params = {
        "dataset": "daily-summaries",
        "stations": station,
        "startDate": f"{year}-01-01",
        "endDate": f"{year}-12-31",
        "format": "csv",
        "units": "metric",
        "dataTypes": ",".join(data_types),
    }
    url = "https://www.ncei.noaa.gov/access/services/data/v1?" + urlencode(params)
    content, _ = _fetch_bytes(url)
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(content)
    print(f"  Wrote {out_path}")


def _download_hourly_weather(year: int) -> None:
    """LaGuardia Airport (ASOS station USW00014732) hourly METAR observations,
    fetched month-by-month -- the NCEI access-data-service silently truncates
    a full-year request."""
    out_path = WEATHER_DIR / f"lga_hourly_weather_{year}.csv"
    dataset = "local-climatological-data-v2"
    station = "USW00014732"
    data_types = ["HourlyDryBulbTemperature", "HourlyPrecipitation", "HourlyPresentWeatherType", "REPORT_TYPE"]

    chunks = []
    header = None
    for month in range(1, 13):
        last_day = calendar.monthrange(year, month)[1]
        params = {
            "dataset": dataset,
            "stations": station,
            "startDate": f"{year}-{month:02d}-01",
            "endDate": f"{year}-{month:02d}-{last_day:02d}",
            "format": "csv",
            "units": "metric",
            "dataTypes": ",".join(data_types),
        }
        url = "https://www.ncei.noaa.gov/access/services/data/v1?" + urlencode(params)
        for attempt in range(4):
            try:
                raw, _ = _fetch_bytes(url)
                lines = raw.decode().splitlines()
                break
            except Exception as e:
                if attempt == 3:
                    raise
                print(f"    {year}-{month:02d}: attempt {attempt + 1} failed ({e}), retrying...")
                time.sleep(10 * (attempt + 1))
        if header is None:
            header = lines[0]
        chunks.extend(lines[1:])
        print(f"    {year}-{month:02d}: {len(lines) - 1} rows")

    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join([header] + chunks) + "\n")
    print(f"  Wrote {len(chunks)} rows to {out_path}")


def download_weather(year: int = YEAR) -> None:
    print(f"Downloading {year} weather (Central Park daily + LaGuardia hourly)...")
    _download_daily_weather(year)
    _download_hourly_weather(year)


# NYC Permitted Event Information - Historical, Socrata dataset bkfu-528j:
# https://data.cityofnewyork.us/City-Government/NYC-Permitted-Event-Information-Historical/bkfu-528j
# NOT dataset tvpp-9vvx ("NYC Permitted Event Information") -- that id looks
# like the right dataset by name, but it's a rolling "next ~30 days" feed
# with no history, so a $where for any past year silently returns 0 rows.
SODA_EVENTS_URL = "https://data.cityofnewyork.us/resource/bkfu-528j.json"
EVENTS_COLUMN_MAP = {
    "event_id": "Event ID",
    "event_name": "Event Name",
    "start_date_time": "Start Date/Time",
    "end_date_time": "End Date/Time",
    "event_agency": "Event Agency",
    "event_type": "Event Type",
    "event_borough": "Event Borough",
    "event_location": "Event Location",
    "event_street_side": "Event Street Side",
    "street_closure_type": "Street Closure Type",
    "community_board": "Community Board",
    "police_precinct": "Police Precinct",
    "cemsid": "CEMSID",
}


def download_events(year: int = YEAR) -> None:
    """Pull NYC Permitted Event Information for `year` only, via the public
    SODA API. The full historical export (2008-present) is ~750MB, but every
    consumer of this file (aggregate_distribution_stats.py,
    generate_design_matrix.py) filters to a single year immediately after
    loading it, so there's no reason to pull the rest. Column names and the
    datetime format are remapped here to match the original manually-exported
    export.csv exactly, so nothing downstream needs to change."""
    print(f"Downloading {year} NYC permitted events...")
    where = f"start_date_time >= '{year}-01-01T00:00:00' AND start_date_time < '{year + 1}-01-01T00:00:00'"
    rows = []
    limit = 50_000
    offset = 0
    while True:
        params = {"$where": where, "$order": "event_id", "$limit": limit, "$offset": offset}
        url = f"{SODA_EVENTS_URL}?{urlencode(params)}"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=60) as resp:
            page = json.loads(resp.read())
        rows.extend(page)
        print(f"  fetched {len(rows)} rows so far...")
        if len(page) < limit:
            break
        offset += limit

    df = pd.DataFrame(rows).rename(columns=EVENTS_COLUMN_MAP)
    for col in ("Start Date/Time", "End Date/Time"):
        df[col] = pd.to_datetime(df[col]).dt.strftime("%m/%d/%Y %I:%M:%S %p")
    df = df.reindex(columns=list(EVENTS_COLUMN_MAP.values()))

    out_path = ROOT / "data" / "export.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Wrote {len(df)} events to {out_path}")


MONTHLY_REPORT_URL = "https://www.nyc.gov/assets/tlc/downloads/csv/data_reports_monthly.csv"
MONTHLY_MODE_BY_CLASS = {"Yellow": "yellow", "Green": "green", "FHV - High Volume": "rideshare"}
MONTHLY_START, MONTHLY_END = "2009-01-01", "2025-12-01"


def _monthly_report() -> pd.DataFrame:
    """TLC aggregate report -> tidy (month, mode, trips_per_day), 2010 onward."""
    raw, _ = _fetch_bytes(MONTHLY_REPORT_URL)
    df = pd.read_csv(io.BytesIO(raw), thousands=",")
    df["month"] = pd.to_datetime(df["Month/Year"])
    df["mode"] = df["License Class"].map(MONTHLY_MODE_BY_CLASS)
    df = df.dropna(subset=["mode"])
    df["trips_per_day"] = pd.to_numeric(df["Trips Per Day"], errors="coerce")
    return df[["month", "mode", "trips_per_day"]]


def _parquet_row_count(url: str) -> int:
    """Row count from the parquet footer alone: one range request for the
    8-byte tail (footer length + magic), one for the footer itself."""
    tail, headers = _fetch_bytes(url, {"Range": "bytes=-8"})
    total_size = int(headers["Content-Range"].split("/")[1])
    footer_len = int.from_bytes(tail[:4], "little")
    footer, _ = _fetch_bytes(url, {"Range": f"bytes={total_size - 8 - footer_len}-{total_size - 1}"})
    return pq.read_metadata(io.BytesIO(footer)).num_rows


def _yellow_2009() -> pd.DataFrame:
    """2009 yellow trips/day from raw-file footers (predates the TLC report)."""
    rows = []
    for m in range(1, 13):
        url = f"{TLC_CDN}/yellow_tripdata_2009-{m:02d}.parquet"
        try:
            n = _parquet_row_count(url)
        except Exception as e:
            print(f"  [warn] 2009-{m:02d}: {e}", file=sys.stderr)
            continue
        month = pd.Timestamp(2009, m, 1)
        rows.append({"month": month, "mode": "yellow", "trips_per_day": n / month.days_in_month})
    return pd.DataFrame(rows)


def download_monthly_trip_counts() -> None:
    print("Downloading long-run (2009-2025) monthly trip counts...")
    combined = pd.concat([_monthly_report(), _yellow_2009()], ignore_index=True)
    combined = combined[combined["month"].between(MONTHLY_START, MONTHLY_END)]
    combined["total_trips"] = (
        (combined["trips_per_day"] * combined["month"].dt.days_in_month).round().astype("Int64")
    )
    combined["trips_per_day"] = combined["trips_per_day"].round(1)
    combined = combined.sort_values(["mode", "month"])

    out_path = PROCESSED_DIR / "monthly_trip_counts_2009_2025.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"  Wrote {len(combined)} rows to {out_path}")


DATASETS = {
    "yellow": ("2025 yellow taxi trips (parquet, ~49M rows/yr)", download_yellow),
    "green": ("2025 green taxi trips (parquet, ~0.6M rows/yr)", download_green),
    "fhvhv": ("2025 HVFHV rideshare trips (parquet, ~244M rows/yr)", download_fhvhv),
    "weather": ("2025 weather -- Central Park daily + LGA hourly", download_weather),
    "events": ("2025 NYC permitted events", download_events),
    "monthly-trip-counts": ("Long-run (2009-2025) monthly trip counts by mode", download_monthly_trip_counts),
}

# "external" = everything a grader needs beyond the TLC trip parquet they
# already have -- weather, events, and the long-run monthly counts are all
# small supplementary files nothing else provides.
BUNDLES = {
    "external": ["weather", "events", "monthly-trip-counts"],
    "all": list(DATASETS),
}


def resolve_selection(names: list[str]) -> list[str]:
    resolved = []
    for name in names:
        for n in BUNDLES.get(name, [name]):
            if n not in resolved:
                resolved.append(n)
    return resolved


def prompt_menu() -> list[str]:
    keys = list(DATASETS)
    extra = ["external", "all"]
    all_keys = keys + extra

    print("What would you like to download?\n")
    for i, key in enumerate(keys, 1):
        print(f"  {i}. {key:22s} {DATASETS[key][0]}")
    print(f"  {len(keys) + 1}. {'external':22s} weather + events + monthly-trip-counts "
          f"(what graders need most -- you already have the TLC trip parquet)")
    print(f"  {len(keys) + 2}. {'all':22s} everything above\n")

    choice = input("Enter numbers or names, comma-separated (e.g. '1,3' or 'events,weather'): ").strip()
    tokens = [t.strip() for t in choice.split(",") if t.strip()]

    selection = []
    for t in tokens:
        if t.isdigit() and 1 <= int(t) <= len(all_keys):
            selection.append(all_keys[int(t) - 1])
        elif t in all_keys:
            selection.append(t)
        else:
            print(f"  [warn] unrecognised choice '{t}', skipping")
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=list(DATASETS) + list(BUNDLES),
        help="one or more of: " + ", ".join(list(DATASETS) + list(BUNDLES)) + ". Omit for an interactive menu.",
    )
    args = parser.parse_args()

    selection = args.dataset if args.dataset else prompt_menu()
    if not selection:
        print("Nothing selected, exiting.")
        return

    for name in resolve_selection(selection):
        print(f"\n=== {name} ===")
        DATASETS[name][1]()

    print("\nDone.")


if __name__ == "__main__":
    main()
