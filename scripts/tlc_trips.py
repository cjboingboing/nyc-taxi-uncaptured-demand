"""
Shared trip-loading logic for the 2025 TLC pipeline scripts.

`aggregate_timeseries_and_summaries.py` and `generate_design_matrix.py` each
carried their own copy of "load yellow/green/FHVHV, harmonize to a common
schema, keep only valid trips" -- identical when first written, but two
independent copies of the same logic drift silently over time. That's exactly
what happened: a third copy in the since-retired scripts/legacy/
aggregate_all_data.py was missing the 264/265 zone exclusion below, and
nobody noticed until it was compared line-by-line. Any change to these rules
should be made HERE so every caller picks it up identically.

VALIDITY RULES (2026-08-24 rewrite -- see PROJECT_CONTEXT.md for the prior,
looser version this replaces). A trip is dropped entirely (not just nulled)
if it fails any of these; there is no separate "null the metric, keep the
row" step anymore -- a trip that fails these checks isn't a trip we trust
enough to count as demand at all:

- duration in (60 seconds, 300 minutes) -- upper bound added after outlier
  analysis on the full 2025 population found values (e.g. a 14,880-minute
  "trip") impossible under any real fare-paying ride; see
  notebooks/general_stats.ipynb for the boxplots/histograms behind this cutoff
- distance in the open interval (0.1, 300) miles
- fare (fare_amount / base_passenger_fare) >= $2.50, the minimum base fare
- cost (total charged) <= $500 -- same outlier analysis; a $500 cap is well
  above any legitimate airport/cross-borough fare ($82.6 avg at JFK) but
  excludes clearly corrupted values (e.g. $863,380.37)
- tip is NULLED (trip kept) above $200 rather than dropped -- tip is not a
  demand-model feature, so an implausible tip shouldn't sink an otherwise
  valid trip
- tolls and every surcharge/tax field are non-negative
- taxi only (yellow/green): VendorID in {1, 2}; RatecodeID in {1..6};
  payment_type in {1..6}; passenger_count in {1..5}
- rideshare (FHVHV) passenger_count would be {1..7} (UberXL) per spec, but
  FHVHV trip records have NO passenger_count column at all (confirmed against
  the actual 2025 parquet schema) -- there's nothing to filter, so this rule
  is a no-op for rideshare, not silently applied to some other field.
- pickup AND dropoff zone must be a real zone: LocationIDs 264 ("NV") / 265
  ("NA") are TLC's catch-all "unknown/outside the zone system" codes, not
  places. The originally-requested rule was a lat/lon bounding rectangle
  around the NYC catchment area, but 2025 TLC files have no lat/lon columns
  at all (removed TLC-wide since 2016) -- this zone-ID check is the closest
  real equivalent given what's actually in the data.
- pickup_date in [year-01-01, year-12-31] (corrupted meter/device
  timestamps, seen as far back as 2007)

KNOWN "UNKNOWN, NOT INVALID" CARVE-OUT: a growing share of yellow/green trips
(15.5% of yellow in Jan 2025, 28.1% by June, still 27.8% in Dec; green
tripled from 3.8% to 12.1% across the year) report RatecodeID = NULL,
payment_type = 0, and passenger_count = NULL together -- one coherent block,
~85% VendorID 2 (Curb Mobility), not scattered garbage: median trip_distance
and fare in this block are close to the rest of the file. This looks like a
vendor reporting-pipeline gap, not bad trips. Since the missing share grows
across the year, dropping this block under a strict reading of the
RatecodeID/payment_type/passenger_count rules would fabricate a fake
declining-taxi-demand trend in 2025 that isn't real. So: NULL RatecodeID,
payment_type == 0, and NULL passenger_count are each treated as "unknown,
keep the trip" rather than "invalid, drop it" -- but only exactly those
sentinel values. A trip with an explicit out-of-range value (passenger_count
== 0 or == 6, RatecodeID == 99 "not available", payment_type == 7) still
fails validity and is dropped; those are real signals of a bad record, not
the same reporting gap.
"""

from pathlib import Path

import pandas as pd

from pyspark.sql import functions as F

MIN_TRIP_SECONDS = 60
MAX_TRIP_MINUTES = 300.0        # 5 hours -- see notebooks/general_stats.ipynb outlier analysis
DISTANCE_RANGE = (0.1, 300.0)   # miles, exclusive both ends
MIN_BASE_FARE = 2.5             # dollars
MAX_COST = 500.0                # dollars -- see notebooks/general_stats.ipynb outlier analysis
MAX_PLAUSIBLE_TIP = 200.0       # dollars -- tip is NULLED above this, trip is kept (not a model feature)

VALID_VENDOR_IDS = (1, 2)
VALID_RATECODE_IDS = tuple(range(1, 7))    # 1..6
VALID_PAYMENT_TYPES = tuple(range(1, 7))   # 1..6
VALID_PASSENGER_COUNT_TAXI = tuple(range(1, 6))  # 1..5

UNKNOWN_ZONE_IDS = (264, 265)

EVENTS_DATETIME_FMT = "%m/%d/%Y %I:%M:%S %p"
# There is a planned event on essentially every day of the year, including
# minor ones (e.g. routine youth-sports permits with light street-closure
# presence) -- restricting to these two types leaves a genuine quiet-vs-busy
# split (real street-closing events only: marathons, parades, etc.) rather
# than "365/365 days have an event".
MAJOR_EVENT_TYPES = ["Parade", "Athletic Race / Tour"]


def load_zones(spark, root: Path, exclude_unknown: bool):
    """Zone lookup (LocationID -> Borough), broadcast for joins against the
    trip fact table.

    `exclude_unknown` is a required argument rather than a hardcoded default
    because at the time this was extracted the two call sites disagreed on
    whether to drop 264/265; now that load_clean_trips itself hard-filters
    trips to real zones (see UNKNOWN_ZONE_IDS above), both call sites should
    pass True- kept as a parameter rather than baked in so that stays an
    explicit, visible choice rather than another hidden assumption.
    """
    zones_sdf = spark.read.csv(
        str(root / "data" / "taxi_zones" / "taxi+_zone_lookup.csv"), header=True, inferSchema=True
    ).select("LocationID", "Borough")
    if exclude_unknown:
        zones_sdf = zones_sdf.where(~F.col("LocationID").isin(*UNKNOWN_ZONE_IDS))
    return F.broadcast(zones_sdf)


def _in_real_zone(pu_col: str, do_col: str):
    return ~F.col(pu_col).isin(*UNKNOWN_ZONE_IDS) & ~F.col(do_col).isin(*UNKNOWN_ZONE_IDS)


def load_major_events(root: Path, year: int) -> pd.DataFrame:
    """NYC permitted events (data/export.csv), filtered to MAJOR events (see
    MAJOR_EVENT_TYPES above) starting in `year`.

    Shared load+parse+filter prefix for aggregate_distribution_stats.py
    (which explodes these into a daily zone-day presence set, clamping
    multi-day spans) and generate_design_matrix.py (which keeps start/end as
    hour-precision timestamps for a between() join, dropping rather than
    clamping a corrupted span) -- each caller does its own downstream
    shaping from here.

    Returns every original column plus parsed `start`/`end` Timestamp
    columns; rows with an unparseable start are dropped, `end` may be NaT.
    """
    events = pd.read_csv(root / "data" / "export.csv", low_memory=False)
    events["start"] = pd.to_datetime(events["Start Date/Time"], format=EVENTS_DATETIME_FMT, errors="coerce")
    events["end"] = pd.to_datetime(events["End Date/Time"], format=EVENTS_DATETIME_FMT, errors="coerce")
    events = events.dropna(subset=["start"])
    events = events[events["start"].dt.year == year]
    events = events[events["Event Type"].isin(MAJOR_EVENT_TYPES)]
    return events


def load_clean_trips(spark, data_dir: Path, year: int):
    """Yellow + green + FHVHV -> one unioned, validity-filtered trip fact
    table. See module docstring for the full rule set.

    Output columns: mode, PULocationID, DOLocationID, pickup_dt, pickup_date,
    pickup_hour, distance, duration_min, cost, tip.

    NOTE: MAX_TRIP_MINUTES and MAX_COST (below) were added after outlier
    analysis on the full 2025 population found impossible values under the
    original bounds -- e.g. cost up to $863,380.37 and duration up to 14,880
    minutes (~10 days) -- neither caught by any rule that existed before. See
    notebooks/general_stats.ipynb for the distributions behind these cutoffs
    and why a mechanical 1.5x-IQR fence was rejected in favour of them (the
    fence itself would flag ordinary long airport fares as outliers).
    """
    yellow_paths = [str(data_dir / f"{year}-{m:02d}.parquet") for m in range(1, 13)]
    green_paths = [str(data_dir / f"green_tripdata_{year}-{m:02d}.parquet") for m in range(1, 13)]
    fhvhv_paths = [str(data_dir / f"fhvhv_tripdata_{year}-{m:02d}.parquet") for m in range(1, 13)]

    yellow_raw = spark.read.parquet(*yellow_paths)
    yellow_duration_sec = F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")
    yellow_valid = (
        F.col("VendorID").isin(*VALID_VENDOR_IDS)
        & (F.col("RatecodeID").isin(*VALID_RATECODE_IDS) | F.col("RatecodeID").isNull())
        & (F.col("payment_type").isin(*VALID_PAYMENT_TYPES) | (F.col("payment_type") == 0))
        & (F.col("passenger_count").isin(*VALID_PASSENGER_COUNT_TAXI) | F.col("passenger_count").isNull())
        & (yellow_duration_sec > MIN_TRIP_SECONDS) & (yellow_duration_sec < MAX_TRIP_MINUTES * 60)
        & (F.col("total_amount") <= MAX_COST)
        & (F.col("trip_distance") > DISTANCE_RANGE[0]) & (F.col("trip_distance") < DISTANCE_RANGE[1])
        & (F.col("fare_amount") >= MIN_BASE_FARE)
        & (F.col("tip_amount") >= 0)
        & (F.col("tolls_amount") >= 0)
        & (F.col("extra") >= 0)
        & (F.col("mta_tax") >= 0)
        & (F.col("improvement_surcharge") >= 0)
        & (F.coalesce(F.col("congestion_surcharge"), F.lit(0.0)) >= 0)
        & (F.coalesce(F.col("Airport_fee"), F.lit(0.0)) >= 0)
        & (F.coalesce(F.col("cbd_congestion_fee"), F.lit(0.0)) >= 0)
        & _in_real_zone("PULocationID", "DOLocationID")
    )
    yellow = (
        yellow_raw.where(yellow_valid)
        .withColumn("mode", F.lit("yellow"))
        .withColumn("pickup_dt", F.col("tpep_pickup_datetime"))
        .withColumn("distance", F.col("trip_distance"))
        .withColumn("duration_min", yellow_duration_sec / 60.0)
        .withColumn("cost", F.col("total_amount"))
        .withColumn("tip", F.col("tip_amount"))
        .select("mode", "PULocationID", "DOLocationID", "pickup_dt", "distance", "duration_min", "cost", "tip")
    )

    green_raw = spark.read.parquet(*green_paths)
    green_duration_sec = F.unix_timestamp("lpep_dropoff_datetime") - F.unix_timestamp("lpep_pickup_datetime")
    green_valid = (
        F.col("VendorID").isin(*VALID_VENDOR_IDS)
        & (F.col("RatecodeID").isin(*VALID_RATECODE_IDS) | F.col("RatecodeID").isNull())
        & (F.col("payment_type").isin(*VALID_PAYMENT_TYPES) | (F.col("payment_type") == 0))
        & (F.col("passenger_count").isin(*VALID_PASSENGER_COUNT_TAXI) | F.col("passenger_count").isNull())
        & (green_duration_sec > MIN_TRIP_SECONDS) & (green_duration_sec < MAX_TRIP_MINUTES * 60)
        & (F.col("total_amount") <= MAX_COST)
        & (F.col("trip_distance") > DISTANCE_RANGE[0]) & (F.col("trip_distance") < DISTANCE_RANGE[1])
        & (F.col("fare_amount") >= MIN_BASE_FARE)
        & (F.col("tip_amount") >= 0)
        & (F.col("tolls_amount") >= 0)
        & (F.col("extra") >= 0)
        & (F.col("mta_tax") >= 0)
        & (F.col("improvement_surcharge") >= 0)
        & (F.coalesce(F.col("congestion_surcharge"), F.lit(0.0)) >= 0)
        & (F.coalesce(F.col("ehail_fee"), F.lit(0.0)) >= 0)
        & (F.coalesce(F.col("cbd_congestion_fee"), F.lit(0.0)) >= 0)
        & _in_real_zone("PULocationID", "DOLocationID")
    )
    green = (
        green_raw.where(green_valid)
        .withColumn("mode", F.lit("green"))
        .withColumn("pickup_dt", F.col("lpep_pickup_datetime"))
        .withColumn("distance", F.col("trip_distance"))
        .withColumn("duration_min", green_duration_sec / 60.0)
        .withColumn("cost", F.col("total_amount"))
        .withColumn("tip", F.col("tip_amount"))
        .select("mode", "PULocationID", "DOLocationID", "pickup_dt", "distance", "duration_min", "cost", "tip")
    )

    # FHVHV has no VendorID / RatecodeID / payment_type / passenger_count
    # columns at all (confirmed against the actual 2025 schema) -- those four
    # rules from the spec simply don't apply here, rather than being mapped
    # onto the wrong field.
    fhvhv_raw = spark.read.parquet(*fhvhv_paths)
    fhvhv_valid = (
        (F.col("trip_time") > MIN_TRIP_SECONDS) & (F.col("trip_time") < MAX_TRIP_MINUTES * 60)
        & (F.col("trip_miles") > DISTANCE_RANGE[0]) & (F.col("trip_miles") < DISTANCE_RANGE[1])
        & (F.col("base_passenger_fare") >= MIN_BASE_FARE)
        & (F.col("tips") >= 0)
        & (F.col("tolls") >= 0)
        & (F.col("bcf") >= 0)
        & (F.col("sales_tax") >= 0)
        & (F.coalesce(F.col("congestion_surcharge"), F.lit(0.0)) >= 0)
        & (F.coalesce(F.col("airport_fee"), F.lit(0.0)) >= 0)
        & (F.coalesce(F.col("cbd_congestion_fee"), F.lit(0.0)) >= 0)
        & _in_real_zone("PULocationID", "DOLocationID")
    )
    fhvhv = (
        fhvhv_raw.where(fhvhv_valid)
        .withColumn(
            "mode",
            F.when(F.col("hvfhs_license_num") == "HV0003", "uber")
            .when(F.col("hvfhs_license_num") == "HV0005", "lyft")
            .otherwise("other_hvfhv"),
        )
        .withColumn("pickup_dt", F.col("pickup_datetime"))
        .withColumn("distance", F.col("trip_miles"))
        .withColumn("duration_min", F.col("trip_time") / 60.0)
        .withColumn(
            "cost",
            F.col("base_passenger_fare")
            + F.coalesce(F.col("tolls"), F.lit(0.0))
            + F.coalesce(F.col("bcf"), F.lit(0.0))
            + F.coalesce(F.col("sales_tax"), F.lit(0.0))
            + F.coalesce(F.col("congestion_surcharge"), F.lit(0.0))
            + F.coalesce(F.col("airport_fee"), F.lit(0.0))
            + F.coalesce(F.col("tips"), F.lit(0.0)),
        )
        .withColumn("tip", F.col("tips"))
        .select("mode", "PULocationID", "DOLocationID", "pickup_dt", "distance", "duration_min", "cost", "tip")
    )

    trips = (
        yellow.unionByName(green).unionByName(fhvhv)
        # FHVHV's `cost` is a constructed sum (no single source column), so the
        # MAX_COST bound is applied here post-union rather than duplicating that
        # formula in fhvhv_valid above; redundant but harmless for yellow/green,
        # which are already bounded via their own `total_amount` check.
        .where(F.col("cost") <= MAX_COST)
        # tip is NOT a demand-model feature -- an implausible tip nulls just that
        # field rather than discarding an otherwise-valid trip (see
        # notebooks/general_stats.ipynb outlier analysis).
        .withColumn("tip", F.when(F.col("tip") > MAX_PLAUSIBLE_TIP, F.lit(None)).otherwise(F.col("tip")))
        .withColumn("pickup_date", F.to_date("pickup_dt"))
        .where((F.col("pickup_date") >= f"{year}-01-01") & (F.col("pickup_date") <= f"{year}-12-31"))
        .withColumn("pickup_hour", F.date_trunc("hour", F.col("pickup_dt")))
    )

    return trips
