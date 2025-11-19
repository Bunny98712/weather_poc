import openmeteo_requests
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
import clickhouse_connect
from datetime import datetime
from tqdm import tqdm
import time


# ----------------------------------------------------------
# CLICKHOUSE CONNECTION
# ----------------------------------------------------------
client = clickhouse_connect.get_client(
    host="localhost",
    port=8123,
    user="admin",
    password="rishu123",
    database="weather_api"
)

print("Connected to ClickHouse")

# ----------------------------------------------------------
# FETCH LIST OF LOCATIONS
# ----------------------------------------------------------
rows = client.query("""
    SELECT state, pincode, latitude, longitude
    FROM state_wise_location
""")

# Convert ClickHouse QueryResult into a raw list for pandas
if hasattr(rows, "result_rows"):
    data = rows.result_rows
elif hasattr(rows, "named_results"):
    data = rows.named_results
elif hasattr(rows, "rows"):
    data = rows.rows
else:
    data = list(rows)

locations = pd.DataFrame(data, columns=["state", "pincode", "latitude", "longitude"])
print(f"Loaded {len(locations)} locations")

# ----------------------------------------------------------
# OPEN-METEO CLIENT SETUP
# ----------------------------------------------------------
cache = requests_cache.CachedSession(".cache", expire_after=3600)
session = retry(cache, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=session)

WEATHER_API = "https://api.open-meteo.com/v1/forecast"
FLOOD_API   = "https://api.open-meteo.com/v1/flood"

# Toggle verbose debug output for troubleshooting (set True to enable)
DEBUG = False
# When True, skip writing to ClickHouse and print the final DataFrame instead
# Set to False to perform live inserts into ClickHouse
DRY_RUN = False

# When debugging, restrict to the first location to keep output focused
if DEBUG:
    locations = locations.head(1)

# ----------------------------------------------------------
# REQUIRED DAILY FIELDS (WEATHER)
# ----------------------------------------------------------
weather_daily_fields = [
    "weather_code", "temperature_2m_max", "temperature_2m_min",
    "apparent_temperature_max", "apparent_temperature_min",
    "sunrise", "sunset", "daylight_duration", "sunshine_duration",
    "uv_index_max", "uv_index_clear_sky_max", "rain_sum",
    "showers_sum", "snowfall_sum", "precipitation_sum",
    "precipitation_hours", "precipitation_probability_max",
    "wind_speed_10m_max", "wind_gusts_10m_max",
    "wind_direction_10m_dominant", "shortwave_radiation_sum",
    "et0_fao_evapotranspiration"
]

# ----------------------------------------------------------
# REQUIRED DAILY FIELDS (FLOOD API)
# ----------------------------------------------------------
flood_daily_fields = [
    "river_discharge",
    "river_discharge_mean",
    "river_discharge_median",
    "river_discharge_max",
    "river_discharge_min",
    "river_discharge_p25",
    "river_discharge_p75"
]


# ----------------------------------------------------------
# PROCESS EACH LOCATION
# ----------------------------------------------------------
for idx, row in tqdm(locations.iterrows(), total=len(locations), desc="Processing locations"):


    state = row["state"]
    pincode = row["pincode"]
    lat = row["latitude"]
    lon = row["longitude"]

    print(f"\nFetching daily weather & flood data for {state} - {pincode}")

    # Build weather API params
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "auto",
        "daily": ",".join(weather_daily_fields),
        # Request wind speeds in km/h to match DB expectations
        "windspeed_unit": "kmh"
    }

    # -----------------------------
    # FETCH WEATHER DAILY DATA
    # -----------------------------
    try:
        resp = session.get(WEATHER_API, params=params)
        resp.raise_for_status()
        wdata = resp.json()
    except Exception as e:
        print(f"Weather API failed for {state}-{pincode}: {e}")
        continue

    daily = wdata.get("daily", {})
    timestamps = pd.to_datetime(daily.get("time", []), utc=True)

    # -----------------------------
    # FETCH FLOOD DAILY DATA
    # -----------------------------
    try:
        # Request flood daily variables explicitly; some locations may not have
        # flood series available and the API may return an empty 'daily'.
        flood_resp = session.get(FLOOD_API, params={
            "latitude": lat,
            "longitude": lon,
            "timezone": "auto",
            "daily": ",".join(flood_daily_fields)
        })
        flood_resp.raise_for_status()
        flood_data = flood_resp.json().get("daily", {})
    except:
        flood_data = {}   # If flood fails, fill with blank arrays

    # Debug: show what keys the flood API returned so we can understand missing columns
    if DEBUG:
        try:
            print("Flood daily keys:", list(flood_data.keys()))
        except Exception:
            pass

    # Normalization helper for matching flood keys
    def normalize(s): return "".join(ch for ch in str(s).lower() if ch.isalnum())

    # Weather arrays
    weather_arrays = [np.asarray(daily.get(f, [])) for f in weather_daily_fields]

    # Flood arrays (match keys safely) — skip if API returned no daily block
    if not flood_data:
        flood_available = False
        if DEBUG:
            print("Flood API returned no daily data; skipping flood fields for this location.")
        flood_arrays = []
    else:
        flood_available = True
        flood_arrays = []
        flood_keys_norm = {k: normalize(k) for k in flood_data.keys()}

        for f in flood_daily_fields:
            target = normalize(f)
            match = None

            # exact match
            for k, kn in flood_keys_norm.items():
                if kn == target:
                    match = k
                    break

            # fallback substring match
            if not match:
                for k, kn in flood_keys_norm.items():
                    if target in kn or kn in target:
                        match = k
                        break

            if match:
                flood_arrays.append(np.asarray(flood_data.get(match, [])))
            else:
                flood_arrays.append(np.array([]))

    # -----------------------------
    # ALIGN ARRAY LENGTHS
    # -----------------------------
    target_len = len(timestamps)
    if target_len == 0:
        print("No daily data timestamps found. Skipping.")
        continue

    all_arrays = weather_arrays + flood_arrays
    aligned = []

    for arr in all_arrays:
        arr = np.atleast_1d(arr)
        l = len(arr)

        if l == target_len:
            aligned.append(arr)
        elif l == 1:
            aligned.append(np.repeat(arr, target_len))
        elif l == 0:
            aligned.append(np.full(target_len, np.nan))
        elif l > target_len:
            aligned.append(arr[:target_len])
        else:
            padding = np.full(target_len - l, np.nan)
            aligned.append(np.concatenate([arr, padding]))

    # -----------------------------
    # BUILD FINAL DATAFRAME
    # -----------------------------
    df = pd.DataFrame({"day": pd.to_datetime(timestamps).date})

    # Add weather fields
    for name, arr in zip(weather_daily_fields, aligned[:len(weather_daily_fields)]):
        df[name] = arr

    # Add flood fields
    for name, arr in zip(flood_daily_fields, aligned[len(weather_daily_fields):]):
        df[name] = arr

    # If the flood API only provided `river_discharge` and not derived stats,
    # populate sensible defaults for the summary columns from `river_discharge`.
    # This explains why you might see values in `river_discharge` but NaN in
    # `river_discharge_mean`, `river_discharge_median`, etc.
    if "river_discharge" in df.columns:
        rd = pd.to_numeric(df["river_discharge"], errors="coerce")

        # Fill each summary column with the river_discharge value when missing
        for col in [
            "river_discharge_mean",
            "river_discharge_median",
            "river_discharge_min",
            "river_discharge_max",
            "river_discharge_p25",
            "river_discharge_p75",
        ]:
            if col not in df.columns or df[col].isna().all():
                df[col] = rd

        # Debug: report if we filled any columns (only when DEBUG=True)
        filled = [c for c in [
            "river_discharge_mean","river_discharge_median",
            "river_discharge_min","river_discharge_max",
            "river_discharge_p25","river_discharge_p75"
        ] if c in df.columns and not df[c].isna().all()]
        if filled and DEBUG:
            print(f"Filled flood summary columns from river_discharge: {filled}")

    # ----------------------------------------------------------
    # FETCH EARQUAKE DATA (USGS)
    # ----------------------------------------------------------
    earthquake_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    # Search recent quakes within a radius; tune params as needed
    eq_params = {
        "format": "geojson",
        "latitude": lat,
        "longitude": lon,
        "maxradiuskm": 10,
        "minmagnitude": 4.5,
        # limit to recent events (last 7 days) to avoid huge responses
        "starttime": (pd.Timestamp.utcnow() - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
        "endtime": pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    }

    try:
        eq_resp = session.get(earthquake_url, params=eq_params)
        eq_resp.raise_for_status()
        eq_json = eq_resp.json()
        features = eq_json.get("features", [])
    except Exception:
        features = []

    # Whether earthquake data is available for this location
    earthquake_available = bool(features)

    # If no earthquakes found -> fill all with NaN/None
    if not features:
        # Use None for missing properties to avoid numeric overflows when
        # pandas coerces types; we'll explicitly coerce/convert afterwards.
        eq_props = {
            "mag": None,
            "place": None,
            "time": None,
            "updated": None,
            "felt": None,
            "cdi": None,
            "mmi": None,
            "alert": None,
            "status": None,
            "tsunami": None,
            "sig": None
        }
    else:
        # Pick the first returned feature (USGS usually orders by time); you can
        # replace this logic with distance-based selection if needed.
        eq_props = features[0].get("properties", {})

    # Normalize earthquake fields with prefix 'earthquake_'
    earthquake_fields = {
        "earthquake_mag": eq_props.get("mag"),
        "earthquake_place": eq_props.get("place"),
        "earthquake_time": eq_props.get("time"),
        "earthquake_updated": eq_props.get("updated"),
        "earthquake_felt": eq_props.get("felt"),
        "earthquake_cdi": eq_props.get("cdi"),
        "earthquake_mmi": eq_props.get("mmi"),
        "earthquake_alert": eq_props.get("alert"),
        "earthquake_status": eq_props.get("status"),
        "earthquake_tsunami": eq_props.get("tsunami"),
        "earthquake_sig": eq_props.get("sig")
    }

    if DEBUG:
        try:
            print(f"Found {len(features)} earthquake features; using props: {list(earthquake_fields.keys())}")
        except Exception:
            pass

    # Broadcast earthquake fields into the dataframe (same value for each day row)
    for k, v in earthquake_fields.items():
        df[k] = v

    # (availability flags removed - not storing flood/earthquake availability)

    # Convert earthquake time fields (USGS returns epoch milliseconds) to datetimes
    if "earthquake_time" in df:
        df["earthquake_time"] = pd.to_datetime(df["earthquake_time"], unit="ms", utc=True, errors="coerce")
    if "earthquake_updated" in df:
        df["earthquake_updated"] = pd.to_datetime(df["earthquake_updated"], unit="ms", utc=True, errors="coerce")


    # Coerce numeric earthquake fields where applicable
    for col in ["earthquake_mag", "earthquake_felt", "earthquake_cdi", "earthquake_mmi", "earthquake_tsunami", "earthquake_sig"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")


    # Add location metadata
    df["state"] = state
    df["pincode"] = pincode
    df["latitude"] = lat
    df["longitude"] = lon

    # Use only today’s row
    today = pd.Timestamp.utcnow().date()
    df = df[df["day"] == today]

    if df.empty:
        print("No current-day rows. Skipping.")
        continue

    # Convert sunrise/sunset to datetime
    for col in ["sunrise", "sunset"]:
        if col in df:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # Convert numeric columns properly (coerce invalid -> NaN)
    # Do not coerce these columns to numeric (they are dates/strings)
    skip = {"day", "state", "pincode", "latitude", "longitude", "sunrise", "sunset", "wind_direction_10m_dominant", "earthquake_time", "earthquake_updated", "earthquake_place", "earthquake_alert", "earthquake_status"}

    for col in df.columns:
        if col in skip:
            continue
        # coerce to numeric where possible; leave NaN for missing/invalid
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure wind direction is string
    if "wind_direction_10m_dominant" in df:
        df["wind_direction_10m_dominant"] = df["wind_direction_10m_dominant"].astype(str)

    # Optional: in-code mapping of column units (useful for logging or docs)
    COLUMN_UNITS = {
        "weather_code": "WMO code",
        "temperature_2m_max": "°C",
        "temperature_2m_min": "°C",
        "apparent_temperature_max": "°C",
        "apparent_temperature_min": "°C",
        "sunrise": "ISO8601",
        "sunset": "ISO8601",
        "daylight_duration": "s",
        "sunshine_duration": "h",
        "uv_index_max": "index",
        "uv_index_clear_sky_max": "index",
        "rain_sum": "mm",
        "showers_sum": "mm",
        "snowfall_sum": "cm",
        "precipitation_sum": "mm",
        "precipitation_hours": "h",
        "precipitation_probability_max": "%",
        "wind_speed_10m_max": "km/h",
        "wind_gusts_10m_max": "km/h",
        "wind_direction_10m_dominant": "°",
        "shortwave_radiation_sum": "MJ/m^2",
        "et0_fao_evapotranspiration": "mm",
        "river_discharge": "m^3/s",
        "river_discharge_mean": "m^3/s",
        "river_discharge_median": "m^3/s",
        "river_discharge_max": "m^3/s",
        "river_discharge_min": "m^3/s",
        "river_discharge_p25": "m^3/s",
        "river_discharge_p75": "m^3/s",
    }

    # Add earthquake column units
    COLUMN_UNITS.update({
        "earthquake_mag": "magnitude",
        "earthquake_place": "text",
        "earthquake_time": "ISO8601",
        "earthquake_updated": "ISO8601",
        "earthquake_felt": "count",
        "earthquake_cdi": "intensity_index",
        "earthquake_mmi": "intensity_index",
        "earthquake_alert": "text",
        "earthquake_status": "text",
        "earthquake_tsunami": "flag",
        "earthquake_sig": "significance",
    })

    # availability flags intentionally omitted from units mapping

    # ----------------------------------------------------------
    # DUPLICATE CHECK (avoid inserting same day multiple times)
    # ----------------------------------------------------------
    q = f"""
        SELECT day FROM weather_daily
        WHERE state = '{state.replace("'", "''")}'
        AND pincode = '{str(pincode)}'
        AND day = toDate('{today}')
    """

    existing = client.query(q)

    # Robustly inspect QueryResult for any returned rows  
    if hasattr(existing, 'rows'):
        existing_iterator = existing.rows
    elif hasattr(existing, 'result_rows'):
        existing_iterator = existing.result_rows
    elif hasattr(existing, 'named_results'):
        existing_iterator = existing.named_results
    elif hasattr(existing, 'data'):
        existing_iterator = existing.data

    has_existing = False
    for r in existing_iterator:
        # if there's at least one row, treat as duplicate
        try:
            if r:
                has_existing = True
                break
        except Exception:
            continue

    if has_existing:
        print(f"Duplicate found for {state}-{pincode}, skipping insert.")
        continue

    # ----------------------------------------------------------
    # INSERT INTO CLICKHOUSE
    # ----------------------------------------------------------
    # Reorder columns to match DDL: state, pincode, latitude, longitude, day, <metrics...>
    ordered_cols = [
        "state", "pincode", "latitude", "longitude", "day"
    ] + weather_daily_fields + flood_daily_fields

    # Keep only columns that exist both in df and in the ClickHouse table.
    # Query ClickHouse system.columns to discover the table's existing columns
    try:
        cols_q = client.query("SELECT name FROM system.columns WHERE database = 'weather_api' AND table = 'weather_daily'")
        if hasattr(cols_q, 'named_results'):
            existing_cols = {r['name'] for r in cols_q.named_results}
        elif hasattr(cols_q, 'result_rows'):
            existing_cols = {r[0] for r in cols_q.result_rows}
        elif hasattr(cols_q, 'rows'):
            existing_cols = {r[0] for r in cols_q.rows}
        else:
            existing_cols = set()
    except Exception:
        existing_cols = set()

    # Compose candidate optional columns (flood/earthquake & flags)
    optional_cols = [
        "river_discharge","river_discharge_mean","river_discharge_median",
        "river_discharge_max","river_discharge_min","river_discharge_p25","river_discharge_p75",
        "earthquake_mag","earthquake_place","earthquake_time","earthquake_updated",
        "earthquake_felt","earthquake_cdi","earthquake_mmi","earthquake_alert",
        "earthquake_status","earthquake_tsunami","earthquake_sig"
    ]

    ordered_cols = ordered_cols + [c for c in optional_cols if c in df.columns]

    # Final filter: only keep columns present in the DataFrame AND the ClickHouse table
    if existing_cols:
        ordered_cols = [c for c in ordered_cols if c in df.columns and c in existing_cols]
    else:
        ordered_cols = [c for c in ordered_cols if c in df.columns]

    # Additionally fetch column types so we can handle non-nullable String columns
    # (ClickHouse will raise if we attempt to insert None into a non-nullable String)
    try:
        type_q = client.query("SELECT name, type FROM system.columns WHERE database = 'weather_api' AND table = 'weather_daily'")
        if hasattr(type_q, 'named_results'):
            col_types = {r['name']: r['type'] for r in type_q.named_results}
        elif hasattr(type_q, 'result_rows'):
            col_types = {r[0]: r[1] for r in type_q.result_rows}
        elif hasattr(type_q, 'rows'):
            col_types = {r[0]: r[1] for r in type_q.rows}
        else:
            col_types = {}
    except Exception:
        col_types = {}

    # Debug: show detected column types (helpful if mapping fails)
    if DEBUG:
        try:
            print("DEBUG: col_types:", list(col_types.items())[:20])
        except Exception:
            pass

    # For non-nullable String columns, replace None/NaN with empty string to avoid insert errors
    # treat any String-like type (including LowCardinality(String), FixedString, etc.)
    # as string, but skip Nullable(...) variants
    non_nullable_string_cols = [
        n for n, t in col_types.items()
        if isinstance(t, str) and 'String' in t and 'Nullable' not in t and n in ordered_cols
    ]
    if DEBUG:
        print("DEBUG: non_nullable_string_cols:", non_nullable_string_cols)
    for col in non_nullable_string_cols:
        if col in df.columns:
            # replace NaN/None with empty string and ensure string/object dtype
            df[col] = df[col].apply(lambda x: '' if pd.isna(x) else x).astype(str)

    df = df[ordered_cols]

    if DRY_RUN:
        print(f"DRY RUN: would insert {len(df)} rows for {state}-{pincode}")
        try:
            print(df.head(5).to_string())
            print("DF shape:", df.shape)
            print("dtypes:\n", df.dtypes)
        except Exception:
            print(df.head(5))
    else:
        # Debug-inspect potential problematic columns before insert
        try:
            if 'earthquake_place' in df.columns:
                val = df['earthquake_place'].iloc[0]
                print("DEBUG: earthquake_place repr:", repr(val), "type:", type(val))
        except Exception as _e:
            print("DEBUG inspect error:", _e)

        client.insert_df("weather_daily", df)
        print(f"Inserted {len(df)} rows for {state}-{pincode}")

# ---- 10 second delay ----
        print("Waiting 30 seconds before next location...\n")
        time.sleep(10)


