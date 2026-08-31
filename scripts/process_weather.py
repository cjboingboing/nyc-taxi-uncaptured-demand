"""
Build clean hourly + daily weather tables and join them with trip volume, for
the weather deep-dive (temperature/precip correlations, hour-of-day-controlled
rain comparison). Pandas-only -- inputs are all small, pre-aggregated CSVs.

Inputs:
- data/weather/lga_hourly_weather_2025.csv   (raw NCEI LCD pull, mixed report types)
- data/weather/central_park_weather_2025.csv (daily summary: precip/snow totals)
- data/processed/intraday_15min_by_date_mode_2025.csv (trip counts per 15-min bucket)

Outputs:
- data/processed/hourly_trips_weather_2025.csv
- data/processed/daily_trips_weather_2025.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WEATHER_DIR = ROOT / "data" / "weather"
PROC_DIR = ROOT / "data" / "processed"

MODE_ORDER = ["yellow", "green", "uber", "lyft"]

# ---------------------------------------------------------------------------
# Hourly weather: keep only standard hourly METAR reports (FM-15), which are
# issued roughly once an hour year-round. generate_design_matrix.py's weather
# join deliberately also keeps FM-16 "special" reports (fired on the hours
# weather is actually changing) -- that's an intentional divergence, not
# drift; see that module's comment for why. The trace-precipitation value
# below (0.01mm) matches generate_design_matrix.py's.
# ---------------------------------------------------------------------------
hourly_wx = pd.read_csv(WEATHER_DIR / "lga_hourly_weather_2025.csv")
hourly_wx = hourly_wx[hourly_wx["REPORT_TYPE"].str.strip() == "FM-15"].copy()
hourly_wx["HourlyDryBulbTemperature"] = pd.to_numeric(hourly_wx["HourlyDryBulbTemperature"], errors="coerce")
hourly_wx["datetime"] = pd.to_datetime(hourly_wx["DATE"])
hourly_wx["date"] = hourly_wx["datetime"].dt.date
hourly_wx["hour"] = hourly_wx["datetime"].dt.hour

# "T" = trace precipitation (present but < 0.01mm) -> treat as a tiny nonzero
# amount so it still counts as "raining" without inflating the mm total
hourly_wx["precip_mm"] = pd.to_numeric(hourly_wx["HourlyPrecipitation"], errors="coerce")
trace_mask = hourly_wx["HourlyPrecipitation"].astype(str).str.strip() == "T"
hourly_wx.loc[trace_mask, "precip_mm"] = 0.01

weather_type = hourly_wx["HourlyPresentWeatherType"].fillna("")
hourly_wx["is_raining"] = weather_type.str.contains("RA|DZ|TS|SH", regex=True) | (hourly_wx["precip_mm"] > 0)
hourly_wx["is_snowing"] = weather_type.str.contains("SN|SG|IC|PL", regex=True)

# a handful of hours have >1 FM-15 report or a gap; collapse to one row/hour
hourly_wx_clean = (
    hourly_wx.groupby(["date", "hour"])
    .agg(
        temp_c=("HourlyDryBulbTemperature", "mean"),
        precip_mm=("precip_mm", "sum"),
        is_raining=("is_raining", "max"),
        is_snowing=("is_snowing", "max"),
    )
    .reset_index()
)
hourly_wx_clean["date"] = pd.to_datetime(hourly_wx_clean["date"])

print(f"Hourly weather: {len(hourly_wx_clean)} / 8760 hours covered in 2025")
print(f"Hours with rain: {hourly_wx_clean['is_raining'].sum()}, with snow: {hourly_wx_clean['is_snowing'].sum()}")

# ---------------------------------------------------------------------------
# Hourly trip counts, from the 15-min intraday buckets (bucket_idx 0..95 -> hour = bucket_idx // 4)
# ---------------------------------------------------------------------------
intraday = pd.read_csv(PROC_DIR / "intraday_15min_by_date_mode_2025.csv", parse_dates=["pickup_date"])
intraday["hour"] = intraday["bucket_idx"] // 4
hourly_trips = (
    intraday.groupby(["pickup_date", "hour", "mode"])["trip_count"].sum().reset_index()
    .rename(columns={"pickup_date": "date"})
)

hourly_merged = hourly_trips.merge(hourly_wx_clean, on=["date", "hour"], how="left")
hourly_merged.to_csv(PROC_DIR / "hourly_trips_weather_2025.csv", index=False)
print(f"Wrote hourly_trips_weather_2025.csv ({len(hourly_merged)} rows)")

# ---------------------------------------------------------------------------
# Daily weather: daily precip/snow TOTALS come from the Central Park daily
# summary (a true daily total, not a sum of possibly-gappy hourly readings);
# daily mean temp comes from the LGA hourly series (Central Park's TAVG field
# is empty for all of 2025 in the GHCN-D feed).
# ---------------------------------------------------------------------------
daily_wx_cp = pd.read_csv(WEATHER_DIR / "central_park_weather_2025.csv", parse_dates=["DATE"])
daily_wx_cp = daily_wx_cp.rename(columns={"DATE": "date", "PRCP": "precip_mm_total", "SNOW": "snow_mm_total"})
# Central Park's TAVG field is empty for all of 2025 in the GHCN-D feed, but
# TMAX/TMIN cover the full year -> use their midpoint as the full-year daily
# temp (the LGA hourly mean is more precise but only covers Jan-Aug, so it's
# used as a within-that-window cross-check, not the primary daily series).
daily_wx_cp["temp_c_mean"] = (daily_wx_cp["TMAX"] + daily_wx_cp["TMIN"]) / 2

daily_temp_hourly = hourly_wx_clean.groupby("date")["temp_c"].mean().rename("temp_c_mean_hourly").reset_index()

daily_wx = daily_wx_cp[["date", "precip_mm_total", "snow_mm_total", "temp_c_mean"]].merge(
    daily_temp_hourly, on="date", how="left"
)
daily_wx["is_rain_day"] = daily_wx["precip_mm_total"].fillna(0) > 0
daily_wx["is_snow_day"] = daily_wx["snow_mm_total"].fillna(0) > 0

corr_check = daily_wx.dropna(subset=["temp_c_mean_hourly"])
print(
    f"Cross-check, Jan-Aug overlap: CP (TMAX+TMIN)/2 vs LGA hourly mean temp, "
    f"corr={corr_check['temp_c_mean'].corr(corr_check['temp_c_mean_hourly']):.3f}, n={len(corr_check)}"
)

daily_wx.to_csv(PROC_DIR / "daily_weather_2025.csv", index=False)
print(f"Wrote daily_weather_2025.csv ({len(daily_wx)} rows)")
