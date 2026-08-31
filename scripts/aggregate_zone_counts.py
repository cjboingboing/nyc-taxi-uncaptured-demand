"""
Aggregate 2025 yellow taxi + green taxi + HVFHV (rideshare) trips by pickup zone.

Produces one small CSV with, per PULocationID:
    - yellow_count       : yellow taxi pickups
    - green_count        : green taxi ("boro taxi") pickups
    - taxi_count         : yellow_count + green_count
    - rideshare_count    : HVFHV (Uber/Lyft/other) pickups
    - total_count        : taxi_count + rideshare_count

Counts are over VALID trips only (tlc_trips.load_clean_trips), so this map
data reflects the same trip population as the demand model rather than every
raw record TLC published -- see that module's docstring for the full rule set.

Green taxis are included alongside yellow on the "taxi" side of the
comparison: they were created specifically to serve upper Manhattan and the
outer boroughs that yellow taxis are largely excluded from picking up in, so
they're a real taxi-side competitor to rideshare in exactly the zones where a
yellow-only view shows near-zero taxi presence. Their 2025 volume is tiny
next to yellow/rideshare but they're kept as a separate column throughout so
the breakdown is visible.

This avoids re-reading the full raw parquet (yellow: ~49M rows/yr,
green: ~0.6M rows/yr, HVFHV: ~244M rows/yr) every time we want to iterate on
the maps.
"""

from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from tlc_trips import load_clean_trips

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "tlc_data"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR = 2025

spark = (
    SparkSession.builder.appName("MAST30034 Project - Zone Aggregation")
    .config("spark.sql.session.timeZone", "Etc/UTC")
    .getOrCreate()
)

# License breakdown is deliberately on RAW, unfiltered FHVHV data (not
# load_clean_trips' output): it's a sanity check on which dispatch services
# are present in the file, not a demand count, so it shouldn't be affected by
# trip-validity filtering.
fhvhv_paths = [str(DATA_DIR / f"fhvhv_tripdata_{YEAR}-{m:02d}.parquet") for m in range(1, 13)]
fhvhv_raw = spark.read.parquet(*fhvhv_paths)
license_counts = (
    fhvhv_raw.groupBy("hvfhs_license_num")
    .agg(F.count("*").alias("trip_count"))
    .orderBy(F.desc("trip_count"))
)
license_counts.show()
license_counts.toPandas().to_csv(OUT_DIR / "hvfhs_license_breakdown_2025.csv", index=False)

# --- Zone pickup counts, over valid trips only ---
trips = load_clean_trips(spark, DATA_DIR, YEAR)

mode_counts = (
    trips.groupBy("PULocationID")
    .pivot("mode", ["yellow", "green", "uber", "lyft", "other_hvfhv"])
    .agg(F.count("*"))
    .fillna(0)
)

zone_counts = (
    mode_counts
    .withColumnRenamed("yellow", "yellow_count")
    .withColumnRenamed("green", "green_count")
    .withColumn("rideshare_count", F.col("uber") + F.col("lyft") + F.col("other_hvfhv"))
    .withColumn("taxi_count", F.col("yellow_count") + F.col("green_count"))
    .withColumn("total_count", F.col("taxi_count") + F.col("rideshare_count"))
    .select("PULocationID", "yellow_count", "green_count", "taxi_count", "rideshare_count", "total_count")
    .orderBy(F.desc("total_count"))
)

zone_counts_pd = zone_counts.toPandas()
out_path = OUT_DIR / "zone_pickup_counts_2025.csv"
zone_counts_pd.to_csv(out_path, index=False)

print(f"Wrote {len(zone_counts_pd)} zones to {out_path}")
print(zone_counts_pd.head(10))
print("Total yellow:", int(zone_counts_pd["yellow_count"].sum()))
print("Total green:", int(zone_counts_pd["green_count"].sum()))
print("Total rideshare:", int(zone_counts_pd["rideshare_count"].sum()))

spark.stop()
