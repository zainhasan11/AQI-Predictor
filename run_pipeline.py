import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
import numpy as np
import duckdb
import joblib
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

print("--- Starting Automated AQI Pipeline Run ---")

# 1. Fetch 1 Year Backfill + Live Data
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

LATITUDE = 33.6844   # Islamabad
LONGITUDE = 73.0479

today = datetime.now().date()
start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')
end_date = today.strftime('%Y-%m-%d')

aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
aq_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi"]
}

aq_response = openmeteo.weather_api(aq_url, params=aq_params)[0]
aq_hourly = aq_response.Hourly()

time_stamps = pd.date_range(
    start=pd.to_datetime(aq_hourly.Time(), unit="s", utc=True),
    end=pd.to_datetime(aq_hourly.TimeEnd(), unit="s", utc=True),
    freq=pd.Timedelta(seconds=aq_hourly.Interval()),
    inclusive="left"
)

df_aq = pd.DataFrame({
    "date_time": time_stamps,
    "pm10": aq_hourly.Variables(0).ValuesAsNumpy(),
    "pm2_5": aq_hourly.Variables(1).ValuesAsNumpy(),
    "co": aq_hourly.Variables(2).ValuesAsNumpy(),
    "no2": aq_hourly.Variables(3).ValuesAsNumpy(),
    "so2": aq_hourly.Variables(4).ValuesAsNumpy(),
    "ozone": aq_hourly.Variables(5).ValuesAsNumpy(),
    "us_aqi": aq_hourly.Variables(6).ValuesAsNumpy()
})

# 2. Feature Engineering
df_features = df_aq.copy().sort_values("date_time").reset_index(drop=True)
df_features["hour"] = df_features["date_time"].dt.hour
df_features["day_of_week"] = df_features["date_time"].dt.dayofweek
df_features["month"] = df_features["date_time"].dt.month

df_features["aqi_lag_1h"] = df_features["us_aqi"].shift(1)
df_features["aqi_lag_3h"] = df_features["us_aqi"].shift(3)
df_features["aqi_change_rate"] = df_features["us_aqi"] - df_features["aqi_lag_1h"]
df_features["target_aqi_24h"] = df_features["us_aqi"].shift(-24)

df_clean = df_features.dropna().reset_index(drop=True)

# 3. Update Feature Store
conn = duckdb.connect('feature_store.duckdb')
conn.execute("CREATE OR REPLACE TABLE aqi_features AS SELECT * FROM df_clean")
conn.close()
print("Updated Feature Store in DuckDB.")

# 4. Retrain Model
feature_cols = [
    'pm10', 'pm2_5', 'co', 'no2', 'so2', 'ozone', 'us_aqi',
    'hour', 'day_of_week', 'month', 'aqi_lag_1h', 'aqi_lag_3h', 'aqi_change_rate'
]
X = df_clean[feature_cols]
y = df_clean['target_aqi_24h']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

ridge_model = Ridge()
ridge_model.fit(X_train, y_train)

# 5. Save Model to Registry
os.makedirs("model_registry", exist_ok=True)
joblib.dump(ridge_model, "model_registry/aqi_model.pkl")
print("Retrained & Saved Model to 'model_registry/aqi_model.pkl'.")
print("--- Pipeline Run Completed Successfully ---")