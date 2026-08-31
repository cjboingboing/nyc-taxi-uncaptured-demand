from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

from tlc_trips import load_clean_trips, load_major_events, load_zones


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "tlc_data"
WEATHER_DIR = ROOT / "data" / "weather"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR = 2025

# Major events (Parade / Athletic Race or Tour) starting in YEAR -- see
# tlc_trips.load_major_events for why only major events. This is the same
# source data aggregate_distribution_stats.py loads, exploded there into a
# daily zone-day presence set rather than kept as hour-precision timestamps.
events = load_major_events(ROOT, YEAR)

# Drop events with a corrupted span (end before start); NaT ends are left in
# as-is -- they simply never match any hour in the between() join below.
bad_event_ids = events.loc[events["end"] < events["start"], "Event ID"]
events = events[~events["Event ID"].isin(bad_event_ids)]

# Keep Event ID too -- needed later to count *distinct* concurrent events
# per (zone, hour) rather than just counting overlap rows.
events = events[["Event ID", "start", "end", "Event Borough", "Event Type"]]

# ============================================================
# Spark session
# ============================================================
spark = (
    SparkSession.builder.appName("MAST30034 Project 1 -- Model data frame")
    .config("spark.sql.session.timeZone", "Etc/UTC")
    .config("spark.sql.shuffle.partitions", "48")
    .config("spark.driver.memory", "8g")
    .getOrCreate()
)

# ------------------------------------------------------------
# Small reference tables -- broadcast these so joins against the huge
# `trips` table (and each other) don't trigger a shuffle.
# ------------------------------------------------------------
# LocationIDs 264 ("NV") and 265 ("NA") are not places -- they're the TLC's
# catch-all codes for trips whose pickup zone is unknown/unrecorded or
# outside the zone system. They have no polygon, borough "Unknown" (so the
# event join can never match), and no operational meaning (cabs can't be
# dispatched to "Unknown"), so they're excluded from the modelling universe.
# Any trips coded there are unattributable to a real zone and drop out with
# them (2025 aggregates show ~0 such pickups anyway).
zones_sdf = load_zones(spark, ROOT, exclude_unknown=True)

# `events` is the cleaned pandas frame from earlier cells (major events only,
# already filtered to YEAR). Still tiny (a few thousand rows) -> broadcast.
event_sdf = F.broadcast(spark.createDataFrame(events))

# ------------------------------------------------------------
# Scaffold: every (zone, hour) combination that *should* exist, whether or
# not a trip was ever recorded there. Left-joining onto this later turns
# "no trips happened" into an explicit trip_count = 0 instead of a missing
# row -- the Poisson model needs those zero-demand zone-hours as real data
# points, not gaps. The two markets are WIDE (taxi_* / rideshare_* columns
# on the same row, via the pivot below) rather than stacked as rows: the
# modelling targets are total_trip_count and taxi_trip_count per zone-hour,
# and the deployable lag features are taxi-only -- one row per (zone, hour)
# means every target and every feature for a cell lives on the same row.
# It also guarantees the row sequence per zone is one-per-hour with no
# gaps, which is what makes the window lags below mean "k hours earlier".
# ------------------------------------------------------------
zones = zones_sdf.select("LocationID").distinct()

hours = spark.range(1).select(
    F.explode(F.sequence(
        F.to_timestamp(F.lit(f"{YEAR}-01-01 00:00:00")),
        F.to_timestamp(F.lit(f"{YEAR}-12-31 23:00:00")),
        F.expr("interval 1 hour"),
    )).alias("pickup_hour")
)

scaffold = zones.crossJoin(hours)

# ------------------------------------------------------------
# Raw trip tables -> one unioned `trips` fact table
# ------------------------------------------------------------
# Schema harmonization and the full trip-validity rule set (duration, distance,
# fare floor, non-negative charges, VendorID/RatecodeID/payment_type/
# passenger_count, real-zone check) all live in tlc_trips.load_clean_trips
# (shared with aggregate_timeseries_and_summaries.py -- see that module's
# docstring for the complete rule list and why this used to be copy-pasted).
trips = load_clean_trips(spark, DATA_DIR, YEAR)

# ------------------------------------------------------------
# Trip-level -> (zone, hour, market) aggregation
# ------------------------------------------------------------
# We deliberately do NOT join events or weather onto `trips` before
# aggregating. Both are properties of (zone, hour) or just (hour) -- not of
# individual trips -- so joining them onto the (huge) trip-level table first
# would be wasted shuffle work, and for events specifically it would fan a
# trip out into one row per *overlapping* event (a trip during 2 concurrent
# parades would duplicate), silently inflating trip_count. So instead: we
# aggregate trips down to (zone, hour, market) first, and join the small
# event/weather summaries on afterwards at that same grain.
trips_agg = (
    trips
    # collapse the 5 raw `mode` values down to the 2 markets we're modelling
    .withColumn(
        "market",
        F.when(F.col("mode").isin("yellow", "green"), "taxi").otherwise("rideshare"),
    )
    .withColumnRenamed("PULocationID", "LocationID")
    .groupBy("LocationID", "pickup_hour")
    # pivot => wide columns per market: taxi_trip_count, taxi_avg_distance,
    # ..., rideshare_trip_count, ... The rideshare_* attribute columns are
    # kept for analysis/EDA and as inference-model targets only -- they can
    # never be *features* of a deployable model (published months late);
    # the deployable features are the taxi_* (internal-data) columns.
    .pivot("market", ["taxi", "rideshare"])
    .agg(
        F.count(F.lit(1)).alias("trip_count"),  # count("*") is disallowed inside a pivot agg
        F.avg("distance").alias("avg_distance"),
        F.variance("distance").alias("var_distance"),   # sample variance; NULL when trip_count == 1
        F.avg("duration_min").alias("avg_duration"),
        F.variance("duration_min").alias("var_duration"),
        F.avg("cost").alias("avg_cost"),
        F.avg("tip").alias("avg_tip"),
        F.sum("tip").alias("total_tip"),
    )
)

# ------------------------------------------------------------
# Event exposure, at (zone, hour) grain -- computed independently of trips
# ------------------------------------------------------------
# Events are only recorded at borough granularity, so cast each event to
# every zone in its borough via the Borough join (a deliberate fan-out:
# "this event affects the whole borough", not just one zone).
event_sdf_zoned = (
    event_sdf.withColumnRenamed("Event Borough", "Borough").join(zones_sdf, on="Borough")
)

# For each (zone, hour), count how many *distinct* events overlap that hour
# in that zone. countDistinct on Event ID (not a plain row count) matters
# because the borough fan-out above can put more than one row per event
# per zone.
event_hours = (
    event_sdf_zoned
    .join(hours, F.col("pickup_hour").between(F.col("start"), F.col("end")))
    .groupBy("LocationID", "pickup_hour")
    .agg(F.countDistinct("Event ID").alias("event_count"))
)

# ------------------------------------------------------------
# Weather, at (hour) grain only -- single station (LGA), so there's no
# zone key to join on; every zone shares the same hourly reading.
# ------------------------------------------------------------
weather_schema = StructType([
    StructField("STATION", StringType(), True),
    StructField("DATE", TimestampType(), True),
    StructField("REPORT_TYPE", StringType(), True),
    StructField("SOURCE", StringType(), True),
    StructField("HourlyDryBulbTemperature", DoubleType(), True),
    StructField("HourlyPrecipitation", StringType(), True),
    StructField("HourlyPresentWeatherType", StringType(), True),
    StructField("REPORT_TYPE_2", StringType(), True)
])

# The raw LCD file has SEVERAL rows per hour (observed: up to 8) -- routine
# FM-15 METARs at ~:51, FM-16 special reports during changing weather, plus
# FM-12/SOD/SOM summary rows whose values are daily/monthly aggregates, not
# hourly readings. Joining this raw onto the scaffold would fan out zone-hour
# rows (duplicating trip_count observations, and doing so precisely on
# bad-weather hours since those get extra FM-16 reports). So: keep only the
# true observation reports, then collapse to exactly one row per hour.
#
# Deliberately keeps FM-16 as well as FM-15 (unlike process_weather.py's
# EDA-facing hourly join, which is FM-15-only): FM-16 "special" reports fire
# exactly on the hours weather is changing, so dropping them would lose
# precipitation signal precisely where it matters most for a rain feature.
# The trace-precipitation value (0.01mm below) matches process_weather.py.
weather = (
    spark.read.option("header", True)
    .option("timestampFormat", "yyyy-MM-dd'T'HH:mm:ss")
    .schema(weather_schema)
    .csv(str(WEATHER_DIR / "lga_hourly_weather_2025.csv"))
    .withColumnRenamed("HourlyDryBulbTemperature", "temp_c")  # file is metric (degrees C)
    .withColumnRenamed("HourlyPresentWeatherType", "weather_type")
    .where(F.trim(F.col("REPORT_TYPE")).isin("FM-15", "FM-16"))
    .withColumn(
        "precip_mm",
        F.when(F.col("HourlyPrecipitation") == 'T', 0.01)
        .otherwise(F.col("HourlyPrecipitation").cast(DoubleType()))
    )
    .withColumn("pickup_hour", F.date_trunc("hour", F.col("DATE")))
    .groupBy("pickup_hour")
    .agg(
        F.avg("temp_c").alias("temp_c"),
        F.max("precip_mm").alias("precip_mm"),
        F.first("weather_type", ignorenulls=True).alias("weather_type"),
    )
)

# ------------------------------------------------------------
# Final design matrix
# ------------------------------------------------------------
# Left-join the scaffold with each summary table at its own natural grain,
# then zero-fill only the columns where "no match" has a genuine, known
# value of zero:
#   - *_trip_count / *_total_tip: a sum over zero trips is legitimately 0
#   - *_var_distance / *_var_duration: 0 or 1 trips both mean "no observed
#     spread" -- we treat that the same as a variance of 0
#   - event_count: no matching event is legitimately 0 events
# The *_avg_* columns are deliberately NOT zero-filled: for a zone-hour
# with 0 trips, "average trip distance" is genuinely undefined (NULL), not
# 0 -- filling it with 0 would falsely claim trips of zero distance
# happened. Handled at modelling time with the missing-indicator method
# (fill + a shared 1{no trips} dummy; see the architecture note, sec.
# "Undefined covariates") -- the indicator makes the fill value irrelevant
# to the slope estimates, so there is no need to fill anything here.
ZERO_FILL = [
    f"{m}_{c}" for m in ("taxi", "rideshare")
    for c in ("trip_count", "var_distance", "var_duration", "total_tip")
] + ["event_count"]

demand = (
    scaffold
    .join(trips_agg, on=["LocationID", "pickup_hour"], how="left")
    .join(event_hours, on=["LocationID", "pickup_hour"], how="left")
    .join(weather, on="pickup_hour", how="left")
    .fillna(0, subset=ZERO_FILL)
    # the two modelling targets: total (market-size proxy) and taxi-only
    # (already a column; total is their sum since taxi/rideshare partition
    # all trips)
    .withColumn("total_trip_count", F.col("taxi_trip_count") + F.col("rideshare_trip_count"))
)

# ------------------------------------------------------------
# Lag block (the Markovian features for the prediction model)
# ------------------------------------------------------------
# HOW THE LAGS WORK: the scaffold guarantees exactly one row per
# (LocationID, pickup_hour) with no gaps -- every hour of the year exists
# for every zone, zero-demand hours included. So within a window
# partitioned by zone and ordered by hour, "the row k positions back" IS
# "the same zone k hours earlier", and F.lag(col, k) needs no timestamp
# arithmetic. (This is also why the lags are computed AFTER the zero-fill:
# a no-trip hour must lag in as a real 0, not as NULL.)
#
# Only INTERNAL (taxi_*) quantities are lagged -- lagged rideshare counts
# would be undeployable, since competitor data is published months late.
# The lag choices, per the operational story:
#   lag 2   = most recent hour whose records are complete at decision time
#             (1h ingestion latency + 1h decision window; also lets nearly
#             all trips picked up in that hour finish, so the avg_* trip
#             attributes exist)
#   lag 24  = same hour yesterday   } strong seasonal-persistence
#   lag 168 = same hour last week   } predictors, immune to latency
#
# The first 168 hours of the year get NULL lags (nothing to look back at;
# there is deliberately no 2024 data in the scaffold) -- drop those rows at
# modelling time. Lagged avg_* columns are NULL wherever the SOURCE hour
# had zero taxi trips; that is intentional and handled at modelling time
# with the missing-indicator method (see comment above), as are the
# log(max(count,1)) / zero-indicator transforms of the lagged counts --
# transforms are modelling decisions, so this table stores lags raw.
w_zone = Window.partitionBy("LocationID").orderBy("pickup_hour")

for k in (2, 24, 168):
    demand = demand.withColumn(f"taxi_count_lag{k}", F.lag("taxi_trip_count", k).over(w_zone))

for col in ("avg_distance", "avg_duration", "avg_cost", "avg_tip"):
    demand = demand.withColumn(f"taxi_{col}_lag2", F.lag(f"taxi_{col}", 2).over(w_zone))

demand.printSchema()
demand.show(5)

# Persist as parquet (partitioned into a handful of files): keeps dtypes
# and NULLs intact and reads straight back into pandas at modelling /
# sub-sampling time. No CSV until after sub-sampling.
demand.coalesce(8).write.mode("overwrite").parquet(str(OUT_DIR / "demand"))



