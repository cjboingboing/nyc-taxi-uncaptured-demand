"""
Aggregate 2025 yellow taxi + green taxi + HVFHV (rideshare) trips into the
exploratory-analysis tables behind notebooks/general_stats.ipynb and
notebooks/timeseries_and_summaries.ipynb. One pass over the raw data builds
`trips` once (cached below) and every output reuses it -- the raw files are
large (yellow ~49M rows/yr, green ~0.6M rows/yr, HVFHV ~244M rows/yr), so
nothing here re-reads or re-validity-filters them a second time.

INPUT
    - data/tlc_data/     raw yellow/green/HVFHV parquet, read and
                          validity-filtered via tlc_trips.load_clean_trips
                          (see that module's docstring for the full rule set
                          -- duration/distance/fare bounds, non-negative
                          charges, VendorID/RatecodeID/payment_type/
                          passenger_count checks, real-zone check).
    - data/taxi_zones/   zone -> borough lookup, via tlc_trips.load_zones.
    - data/export.csv    NYC event-permit dataset, used to flag pickup
                          (date, borough) pairs with a genuinely disruptive
                          event (Parade / Athletic Race or Tour only --
                          routine permits like farmers markets are excluded,
                          else ~365/365 days "have an event" and there's no
                          quiet baseline to compare against).

OUTPUT -- all eight written as CSV to data/processed/:
    - daily_counts_by_mode_2025.csv        one row per (date, mode):
                                            trip_count. Full-year time series
                                            / "average week" day-of-week
                                            profile downstream in pandas.
    - airport_dropoff_summary_2025.csv     one row per (date, mode, airport):
                                            trip_count + avg distance/
                                            duration/cost/tip, for trips
                                            ENDING at JFK or LGA. Joined with
                                            Central Park weather downstream.
    - event_day_trip_summary_2025.csv      one row per (mode, borough,
                                            is_event_day): trip_count + avg
                                            distance/duration/cost/tip, split
                                            by whether the pickup date+borough
                                            coincides with a major event.
    - intraday_15min_by_date_mode_2025.csv one row per (date, day-of-week,
                                            15-min bucket, mode): trip_count.
                                            Averaged to a single "typical
                                            week" profile downstream, where
                                            the full 96-bucket grid can be
                                            reindexed per date so genuinely-
                                            zero buckets count as 0.
    - mode_summary_stats_2025.csv          one row per mode: count, mean,
                                            stddev, min/25/50/75/max for
                                            distance, duration_min, cost, tip.
                                            The general "what does this
                                            dataset look like" overview.
    - zone_mode_counts_2025.csv            one row per (PULocationID, mode):
                                            trip_count -- reused from
                                            zone_mode_boxplot_stats' own
                                            count rather than a second pass
                                            over the same grouping. Sort/
                                            limit in pandas downstream for
                                            "busiest zone x mode
                                            combinations".
    - zone_mode_boxplot_stats_2025.csv     one row per (PULocationID, mode):
                                            Tukey five-number summary
                                            (whislo, q1, median, q3, whishi)
                                            for each of distance/
                                            duration_min/cost/tip -- feeds
                                            matplotlib boxplots directly via
                                            Axes.bxp(), no raw trip rows ever
                                            collected to the driver.
    - mode_outlier_rates_2025.csv          one row per mode: what share of
                                            that mode's trips fall outside
                                            its own IQR fence, per metric.

There is no literal "fare type" column in load_clean_trips' harmonized schema
(RatecodeID/payment_type are validity-filter inputs only, not carried through)
-- `cost` (total charged) and `tip` stand in as the fare-shaped metrics
alongside `distance` and `duration_min`.

Comparability note: yellow taxi `tip_amount` only captures CREDIT CARD tips
(TLC data dictionary) -- cash tips are not recorded. HVFHV `tips` captures
ALL in-app tips. This makes yellow tip/cost figures a slight underestimate
relative to HVFHV's wherever the two are compared side by side (mode summary,
event, and airport tables above).

Every quantile below uses percentile_approx (a Spark SQL aggregate function,
usable inside .agg()/.groupBy().agg()) rather than DataFrame.approxQuantile
(a driver-side method that only computes over the whole DataFrame per call --
no groupBy support). percentile_approx is what makes "quartiles per zone per
mode" a single distributed pass instead of ~1300 separate driver round trips.
"""

from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DateType

import pandas as pd

from tlc_trips import load_clean_trips, load_major_events, load_zones

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "tlc_data"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR = 2025
METRICS = ["distance", "duration_min", "cost", "tip"]
AIRPORT_IDS = {132: "JFK", 138: "LGA"}

# ---------------------------------------------------------------------------
# 1. Build the (date, borough) event-day set from the events dataset (pandas
#    - it's a flat CSV, easier to parse dates here than in Spark SQL).
# ---------------------------------------------------------------------------
events = load_major_events(ROOT, YEAR)

MAX_EVENT_SPAN_DAYS = 14  # guard against bad data producing huge date ranges

event_day_borough = set()
for _, row in events.iterrows():
    start_date = row["start"].date()
    end_date = row["end"].date() if pd.notna(row["end"]) else start_date
    span = (end_date - start_date).days
    if span < 0 or span > MAX_EVENT_SPAN_DAYS:
        end_date = start_date
    for d in pd.date_range(start_date, end_date, freq="D"):
        event_day_borough.add((d.date().isoformat(), row["Event Borough"]))

event_pd = pd.DataFrame(sorted(event_day_borough), columns=["event_date", "Borough"])
print(f"Distinct (date, borough) pairs with a permitted event in {YEAR}: {len(event_pd)}")

# ---------------------------------------------------------------------------
spark = (
    SparkSession.builder.appName("MAST30034 Project 1 - Exploratory Analysis")
    .config("spark.sql.session.timeZone", "Etc/UTC")
    .config("spark.sql.shuffle.partitions", "48")
    .config("spark.driver.memory", "8g")
    .getOrCreate()
)

event_sdf = spark.createDataFrame(event_pd).withColumn(
    "event_date", F.col("event_date").cast(DateType())
)
event_sdf = F.broadcast(event_sdf)

# Resolved 2026-08-24: this used to disagree with generate_design_matrix.py
# (False vs True) as an artifact of the tlc_trips extraction, not a deliberate
# choice. Now moot either way -- load_clean_trips hard-filters every trip's
# PULocationID/DOLocationID to a real zone, so 264/265 never reach this join.
# Both scripts pass True for consistency.
zones_sdf = load_zones(spark, ROOT, exclude_unknown=True)

# Schema harmonization and the full trip-validity rule set (duration, distance,
# fare floor, non-negative charges, VendorID/RatecodeID/payment_type/
# passenger_count, real-zone check) all live in tlc_trips.load_clean_trips
# (shared with generate_design_matrix.py -- see that module's docstring).
#
# Cached once and reused across all eight aggregation passes below -- avoids
# re-reading + re-validity-filtering the raw parquet on every pass.
trips = load_clean_trips(spark, DATA_DIR, YEAR).cache()
print(f"Valid trips loaded: {trips.count():,}")

# ---------------------------------------------------------------------------
# 1. Daily volume by mode
# ---------------------------------------------------------------------------
daily_counts = (
    trips.groupBy("pickup_date", "mode")
    .agg(F.count("*").alias("trip_count"))
    .orderBy("pickup_date", "mode")
)
daily_counts.toPandas().to_csv(OUT_DIR / "daily_counts_by_mode_2025.csv", index=False)
print("Wrote daily_counts_by_mode_2025.csv")

# ---------------------------------------------------------------------------
# 2. Airport (JFK/LGA) DROPOFF daily summary by mode
# ---------------------------------------------------------------------------
airport_map = F.create_map(*[x for k, v in AIRPORT_IDS.items() for x in (F.lit(k), F.lit(v))])

airport_trips = trips.withColumn("airport", airport_map[F.col("DOLocationID")]).where(
    F.col("airport").isNotNull()
)

airport_summary = (
    airport_trips.groupBy("pickup_date", "mode", "airport")
    .agg(
        F.count("*").alias("trip_count"),
        F.avg("distance").alias("avg_distance_miles"),
        F.avg("duration_min").alias("avg_duration_min"),
        F.avg("cost").alias("avg_cost"),
        F.avg("tip").alias("avg_tip"),
    )
    .orderBy("pickup_date", "mode", "airport")
)
airport_summary.toPandas().to_csv(OUT_DIR / "airport_dropoff_summary_2025.csv", index=False)
print("Wrote airport_dropoff_summary_2025.csv")

# ---------------------------------------------------------------------------
# 3. Event-day vs non-event-day trip summary by mode + borough
# ---------------------------------------------------------------------------
trips_with_borough = trips.join(zones_sdf, trips.PULocationID == zones_sdf.LocationID, "left")

trips_with_event_flag = trips_with_borough.join(
    event_sdf,
    (trips_with_borough.pickup_date == event_sdf.event_date)
    & (trips_with_borough.Borough == event_sdf.Borough),
    "left",
).withColumn("is_event_day", F.col("event_date").isNotNull())

event_summary = (
    trips_with_event_flag.groupBy("mode", trips_with_borough.Borough, "is_event_day")
    .agg(
        F.count("*").alias("trip_count"),
        F.avg("distance").alias("avg_distance_miles"),
        F.avg("duration_min").alias("avg_duration_min"),
        F.avg("cost").alias("avg_cost"),
        F.avg("tip").alias("avg_tip"),
    )
    .orderBy("mode", "Borough", "is_event_day")
)
event_summary.toPandas().to_csv(OUT_DIR / "event_day_trip_summary_2025.csv", index=False)
print("Wrote event_day_trip_summary_2025.csv")

# ---------------------------------------------------------------------------
# 4. Intraday profile: trip counts per 15-minute bucket, per (date, day-of-week,
#    mode). Averaging to a single "typical week" happens downstream in pandas,
#    where we can correctly reindex the full 96-bucket grid per date so that
#    genuinely-zero buckets count as 0 in the average (not just omitted).
# ---------------------------------------------------------------------------
intraday = (
    trips.withColumn("dow", F.date_format("pickup_date", "EEEE"))
    .withColumn("bucket_idx", F.floor((F.hour("pickup_dt") * 60 + F.minute("pickup_dt")) / 15))
    .groupBy("pickup_date", "dow", "bucket_idx", "mode")
    .agg(F.count("*").alias("trip_count"))
)
intraday.toPandas().to_csv(OUT_DIR / "intraday_15min_by_date_mode_2025.csv", index=False)
print("Wrote intraday_15min_by_date_mode_2025.csv")

# ---------------------------------------------------------------------------
# 5. Per-mode overview stats. groupBy("mode") -> one row per mode. The
#    percentile_approx call asks for all 5 quantiles in one array-valued
#    aggregate rather than 5 separate expressions -- one pass either way, but
#    fewer aggregate functions for Spark to plan.
# ---------------------------------------------------------------------------
summary_aggs = [F.count("*").alias("n")]
for m in METRICS:
    summary_aggs += [
        F.avg(m).alias(f"{m}_mean"),
        F.stddev(m).alias(f"{m}_stddev"),
        F.expr(f"percentile_approx({m}, array(0.0, 0.25, 0.5, 0.75, 1.0))").alias(f"{m}_q"),
    ]

mode_summary = trips.groupBy("mode").agg(*summary_aggs)
for m in METRICS:
    mode_summary = (
        mode_summary.withColumn(f"{m}_min", F.col(f"{m}_q")[0])
        .withColumn(f"{m}_q1", F.col(f"{m}_q")[1])
        .withColumn(f"{m}_median", F.col(f"{m}_q")[2])
        .withColumn(f"{m}_q3", F.col(f"{m}_q")[3])
        .withColumn(f"{m}_max", F.col(f"{m}_q")[4])
        .drop(f"{m}_q")
    )
mode_summary.toPandas().to_csv(OUT_DIR / "mode_summary_stats_2025.csv", index=False)
print("Wrote mode_summary_stats_2025.csv")

# ---------------------------------------------------------------------------
# 6. Per (zone, mode) boxplot stats.
#
# Quartiles come from one groupBy+agg pass; Tukey fences (q1 - 1.5*iqr,
# q3 + 1.5*iqr) are then a pure column expression on that tiny result, no
# extra pass over `trips`. Getting the WHISKER ends right (nearest real value
# inside the fence, not the fence itself, and not the raw min/max) needs a
# second pass: broadcast the per-group fences back onto every trip row (cheap
# -- broadcasting ~1300 rows avoids a shuffle entirely) and take a
# fence-conditioned min/max per group.
# ---------------------------------------------------------------------------
quantile_aggs = [F.count("*").alias("n")] + [
    F.expr(f"percentile_approx({m}, array(0.25, 0.5, 0.75))").alias(f"{m}_q") for m in METRICS
]
zone_mode_quartiles = trips.groupBy("PULocationID", "mode").agg(*quantile_aggs)
for m in METRICS:
    zone_mode_quartiles = (
        zone_mode_quartiles.withColumn(f"{m}_q1", F.col(f"{m}_q")[0])
        .withColumn(f"{m}_median", F.col(f"{m}_q")[1])
        .withColumn(f"{m}_q3", F.col(f"{m}_q")[2])
        .withColumn(f"{m}_iqr", F.col(f"{m}_q3") - F.col(f"{m}_q1"))
        .withColumn(f"{m}_lower_fence", F.col(f"{m}_q1") - 1.5 * F.col(f"{m}_iqr"))
        .withColumn(f"{m}_upper_fence", F.col(f"{m}_q3") + 1.5 * F.col(f"{m}_iqr"))
        .drop(f"{m}_q")
    )
zone_mode_quartiles = zone_mode_quartiles.cache()

# ---------------------------------------------------------------------------
# 7. Zone x mode trip counts. Derived from zone_mode_quartiles' "n" column
#    above rather than a second groupBy("PULocationID", "mode") pass over
#    `trips` -- both are counting the exact same groups, so a fresh pass
#    would just recompute a value we already have.
# ---------------------------------------------------------------------------
zone_mode_counts = zone_mode_quartiles.select(
    "PULocationID", "mode", F.col("n").alias("trip_count")
).orderBy(F.desc("trip_count"))
zone_mode_counts.toPandas().to_csv(OUT_DIR / "zone_mode_counts_2025.csv", index=False)
print("Wrote zone_mode_counts_2025.csv")

fence_cols = [f"{m}_lower_fence" for m in METRICS] + [f"{m}_upper_fence" for m in METRICS]
trips_with_fences = trips.join(
    F.broadcast(zone_mode_quartiles.select("PULocationID", "mode", *fence_cols)),
    on=["PULocationID", "mode"],
    how="left",
)

whisker_aggs = []
for m in METRICS:
    whisker_aggs += [
        F.min(F.when(F.col(m) >= F.col(f"{m}_lower_fence"), F.col(m))).alias(f"{m}_whislo"),
        F.max(F.when(F.col(m) <= F.col(f"{m}_upper_fence"), F.col(m))).alias(f"{m}_whishi"),
    ]
zone_mode_whiskers = trips_with_fences.groupBy("PULocationID", "mode").agg(*whisker_aggs)

zone_mode_boxplot_stats = zone_mode_quartiles.join(zone_mode_whiskers, on=["PULocationID", "mode"]).select(
    "PULocationID",
    "mode",
    "n",
    *[c for m in METRICS for c in (f"{m}_whislo", f"{m}_q1", f"{m}_median", f"{m}_q3", f"{m}_whishi")],
)
zone_mode_boxplot_stats.toPandas().to_csv(OUT_DIR / "zone_mode_boxplot_stats_2025.csv", index=False)
print("Wrote zone_mode_boxplot_stats_2025.csv")

# ---------------------------------------------------------------------------
# 8. Per-mode outlier rates -- same IQR-fence idea as step 6, but grouped
#    only by mode (coarser than per-zone) since that's the comparison worth
#    reading off a single bar chart. Flags are computed at row level (so
#    outlier-ness is exact, not itself an approximation) but only the
#    aggregated RATE is written out -- flagging ~290M individual trips would
#    make a CSV, not a summary.
# ---------------------------------------------------------------------------
mode_fence_aggs = [
    F.expr(f"percentile_approx({m}, array(0.25, 0.75))").alias(f"{m}_q") for m in METRICS
]
mode_quartiles = trips.groupBy("mode").agg(*mode_fence_aggs)
for m in METRICS:
    mode_quartiles = (
        mode_quartiles.withColumn(f"{m}_q1", F.col(f"{m}_q")[0])
        .withColumn(f"{m}_q3", F.col(f"{m}_q")[1])
        .withColumn(f"{m}_iqr", F.col(f"{m}_q3") - F.col(f"{m}_q1"))
        .withColumn(f"{m}_lower_fence", F.col(f"{m}_q1") - 1.5 * F.col(f"{m}_iqr"))
        .withColumn(f"{m}_upper_fence", F.col(f"{m}_q3") + 1.5 * F.col(f"{m}_iqr"))
        .drop(f"{m}_q")
    )

trips_with_mode_fences = trips.join(F.broadcast(mode_quartiles), on="mode", how="left")
for m in METRICS:
    trips_with_mode_fences = trips_with_mode_fences.withColumn(
        f"{m}_is_outlier",
        (F.col(m) < F.col(f"{m}_lower_fence")) | (F.col(m) > F.col(f"{m}_upper_fence")),
    )

outlier_rate_aggs = [F.count("*").alias("n")] + [
    F.avg(F.col(f"{m}_is_outlier").cast("double")).alias(f"{m}_outlier_rate") for m in METRICS
]
mode_outlier_rates = trips_with_mode_fences.groupBy("mode").agg(*outlier_rate_aggs)
mode_outlier_rates.toPandas().to_csv(OUT_DIR / "mode_outlier_rates_2025.csv", index=False)
print("Wrote mode_outlier_rates_2025.csv")

spark.stop()
