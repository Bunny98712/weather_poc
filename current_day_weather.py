import openmeteo_requests
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
import clickhouse_connect
from datetime import datetime
from tqdm import tqdm
import time


# ========================================================================
# 1. CLICKHOUSE CONNECTION
# ========================================================================

def get_clickhouse_client():
    return clickhouse_connect.get_client(
        host="localhost",
        port=8123,
        user="admin",
        password="rishu123",
        database="weather_api"
    )


# ========================================================================
# 2. LOAD ALL LOCATIONS
# ========================================================================

def load_locations(client):
    rows = client.query("""
        SELECT state, pincode, latitude, longitude
        FROM state_wise_location
    """)

    # Convert QueryResult safely to list
    if hasattr(rows, "result_rows"):
        data = rows.result_rows
    elif hasattr(rows, "rows"):
        data = rows.rows
    elif hasattr(rows, "named_results"):
        data = rows.named_results
    else:
        data = list(rows)

    return pd.DataFrame(data, columns=["state", "pincode", "latitude", "longitude"])


# ========================================================================
# 3. WEATHER API FETCH
# ========================================================================

def fetch_weather(session, lat, lon, fields):
    WEATHER_API = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "auto",
        "daily": ",".join(fields),
        "windspeed_unit": "kmh"
    }

    try:
        r = session.get(WEATHER_API, params=params)
        r.raise_for_status()
        return r.json().get("daily", {})
    except:
        return {}


# ========================================================================
# 4. FLOOD API FETCH
# ========================================================================

def fetch_flood(session, lat, lon, fields):
    FLOOD_API = "https://api.open-meteo.com/v1/flood"

    try:
        r = session.get(FLOOD_API, params={
            "latitude": lat,
            "longitude": lon,
            "timezone": "auto",
            "daily": ",".join(fields),
        })
        r.raise_for_status()
        return r.json().get("daily", {})
    except:
        return {}


# ========================================================================
# 5. EARTHQUAKE API FETCH (USGS)
# ========================================================================

def fetch_earthquake(session, lat, lon):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    params = {
        "format": "geojson",
        "latitude": lat,
        "longitude": lon,
        "maxradiuskm": 50,
        "minmagnitude": 4.5,
        "starttime": (pd.Timestamp.utcnow() - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
        "endtime": pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    }

    try:
        r = session.get(url, params=params)
        r.raise_for_status()
        features = r.json().get("features", [])
        return features[0]["properties"] if features else {}
    except:
        return {}


# ========================================================================
# 6. ALIGN ARRAYS TO SAME LENGTH
# ========================================================================

def align_arrays(arrays, target_len):
    aligned = []

    for arr in arrays:
        arr = np.atleast_1d(arr)
        l = len(arr)

        if l == target_len:
            aligned.append(arr)
        elif l == 0:
            aligned.append(np.full(target_len, np.nan))
        elif l == 1:
            aligned.append(np.repeat(arr, target_len))
        elif l > target_len:
            aligned.append(arr[:target_len])
        else:
            padding = np.full(target_len - l, np.nan)
            aligned.append(np.concatenate([arr, padding]))

    return aligned


# ========================================================================
# 7. BUILD FINAL DATAFRAME FOR ONE LOCATION
# ========================================================================

def build_dataframe(weather_daily, flood_daily, earthquake, weather_fields, flood_fields, state, pincode, lat, lon):
    timestamps = pd.to_datetime(weather_daily.get("time", []), utc=True)

    if len(timestamps) == 0:
        return pd.DataFrame()

    # weather arrays
    weather_arrays = [np.asarray(weather_daily.get(f, [])) for f in weather_fields]

    # flood arrays
    flood_arrays = [np.asarray(flood_daily.get(f, [])) for f in flood_fields]

    # align all together
    aligned = align_arrays(weather_arrays + flood_arrays, len(timestamps))

    # build df
    df = pd.DataFrame({"day": pd.to_datetime(timestamps).date})

    # fill weather
    for name, arr in zip(weather_fields, aligned[:len(weather_fields)]):
        df[name] = arr

    # fill flood
    for name, arr in zip(flood_fields, aligned[len(weather_fields):]):
        df[name] = arr

    # earthquake (same for all rows)
    for k, v in {
        "earthquake_mag": earthquake.get("mag"),
        "earthquake_place": earthquake.get("place"),
        "earthquake_time": earthquake.get("time"),
        "earthquake_updated": earthquake.get("updated"),
        "earthquake_felt": earthquake.get("felt"),
        "earthquake_cdi": earthquake.get("cdi"),
        "earthquake_mmi": earthquake.get("mmi"),
        "earthquake_alert": earthquake.get("alert"),
        "earthquake_status": earthquake.get("status"),
        "earthquake_tsunami": earthquake.get("tsunami"),
        "earthquake_sig": earthquake.get("sig"),
    }.items():
        df[k] = v

    # Convert earthquake timestamps
    df["earthquake_time"] = pd.to_datetime(df["earthquake_time"], unit="ms", utc=True, errors="coerce")
    df["earthquake_updated"] = pd.to_datetime(df["earthquake_updated"], unit="ms", utc=True, errors="coerce")

    df["state"] = state
    df["pincode"] = pincode
    df["latitude"] = lat
    df["longitude"] = lon

    # keep only today
    today = pd.Timestamp.utcnow().date()
    df = df[df["day"] == today]

    return df


# ========================================================================
# 8. CHECK DUPLICATE
# ========================================================================

def row_exists(client, state, pincode, day):
    q = f"""
        SELECT 1 FROM weather_daily 
        WHERE state = '{state.replace("'", "''")}'
        AND pincode = '{pincode}'
        AND day = toDate('{day}')
        LIMIT 1
    """

    r = client.query(q)
    if hasattr(r, "rows") and len(r.rows) > 0:
        return True
    return False


# ========================================================================
# 9. INSERT DATAFRAME SAFELY
# ========================================================================
def insert_dataframe(client, df):

    if df.empty:
        return False

    # -----------------------------------------------------------
    # FIX: Convert all ClickHouse string columns to proper str
    # -----------------------------------------------------------
    # Get table schema
    col_info = client.query("""
        SELECT name, type 
        FROM system.columns 
        WHERE database = 'weather_api' 
        AND table = 'weather_daily'
    """)

    schema = {}
    for r in col_info.result_rows:
        schema[r[0]] = r[1]

    # Fix all non-nullable string-type columns
    for col, ch_type in schema.items():
        if col not in df.columns:
            continue

        if "String" in ch_type and "Nullable" not in ch_type:
            df[col] = df[col].astype(str).replace("nan", "")

        elif "String" in ch_type and "Nullable" in ch_type:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace("nan", None)

    # -----------------------------------------------------------
    # After fixing, insert safely
    # -----------------------------------------------------------
    client.insert_df("weather_daily", df)
    return True




# ========================================================================
# 10. MAIN DRIVER FUNCTION
# ========================================================================

def main():

    print("Connecting to ClickHouse...")
    client = get_clickhouse_client()

    print("Loading locations...")
    locations = load_locations(client)
    print(f"Loaded {len(locations)} locations.\n")

    # API client
    cache = requests_cache.CachedSession(".cache", expire_after=3600)
    session = retry(cache, retries=5, backoff_factor=0.2)

    weather_fields = [
        "weather_code","temperature_2m_max","temperature_2m_min",
        "apparent_temperature_max","apparent_temperature_min",
        "sunrise","sunset","daylight_duration","sunshine_duration",
        "uv_index_max","uv_index_clear_sky_max","rain_sum","showers_sum",
        "snowfall_sum","precipitation_sum","precipitation_hours",
        "precipitation_probability_max","wind_speed_10m_max",
        "wind_gusts_10m_max","wind_direction_10m_dominant",
        "shortwave_radiation_sum","et0_fao_evapotranspiration"
    ]

    flood_fields = [
        "river_discharge","river_discharge_mean","river_discharge_median",
        "river_discharge_max","river_discharge_min",
        "river_discharge_p25","river_discharge_p75"
    ]

    # Process each location
    for _, row in tqdm(locations.iterrows(), total=len(locations)):
        state = row.state
        pincode = row.pincode
        lat = row.latitude
        lon = row.longitude

        print(f"\nFetching data for {state} - {pincode}")

        weather_daily = fetch_weather(session, lat, lon, weather_fields)
        flood_daily = fetch_flood(session, lat, lon, flood_fields)
        eq = fetch_earthquake(session, lat, lon)

        df = build_dataframe(weather_daily, flood_daily, eq, weather_fields, flood_fields, state, pincode, lat, lon)

        if df.empty:
            print("No today data. Skipping.")
            continue

        today = pd.Timestamp.utcnow().date()

        if row_exists(client, state, pincode, today):
            print("Duplicate found. Skipping insert.")
            continue

        if insert_dataframe(client, df):
            print(f"Inserted {len(df)} rows.")
        else:
            print("Insert failed.")

        print("Waiting 4 sec...\n")
        time.sleep(4)


# ========================================================================
# RUN
# ========================================================================

if __name__ == "__main__":
    main()
