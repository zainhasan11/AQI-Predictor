import streamlit as st
import pandas as pd
import numpy as np
import joblib
import openmeteo_requests
import requests_cache
from retry_requests import retry
import plotly.graph_objects as go
import os

# Page Configuration
st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌫️",
    layout="wide"
)

# Title & Description
st.title("🌫️ Pearls Air Quality Index (AQI) Predictor")
st.markdown("Real-time air quality monitoring and **24-hour forecasting** powered by serverless ML.")

# 1. Load Model
@st.cache_resource
def load_model():
    model_path = "model_registry/aqi_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# 2. Fetch Live Data Function
@st.cache_data(ttl=1800)  # Refresh every 30 mins
def fetch_live_aqi():
    cache_session = requests_cache.CachedSession('.cache', expire_after=1800)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    LATITUDE = 33.6844   # Islamabad
    LONGITUDE = 73.0479

    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi"],
        "forecast_days": 3
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
    return df_aq

# Helper for AQI Category & Colors
def get_aqi_status(aqi):
    if aqi <= 50:
        return "Good", "🟢", "#00e400"
    elif aqi <= 100:
        return "Moderate", "🟡", "#ffff00"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "🟠", "#ff7e00"
    elif aqi <= 200:
        return "Unhealthy", "🔴", "#ff0000"
    elif aqi <= 300:
        return "Very Unhealthy", "🟣", "#8f3f97"
    else:
        return "Hazardous", "🤎", "#7e0023"

# Main Logic
try:
    df_live = fetch_live_aqi()
    current_row = df_live.iloc[0]
    current_aqi = int(current_row['us_aqi'])
    status, emoji, color = get_aqi_status(current_aqi)

    # Top Metrics Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Current US AQI", value=f"{current_aqi} {emoji}")
    with col2:
        st.metric(label="AQI Category", value=status)
    with col3:
        st.metric(label="PM2.5", value=f"{current_row['pm2_5']:.1f} µg/m³")
    with col4:
        st.metric(label="PM10", value=f"{current_row['pm10']:.1f} µg/m³")

    st.divider()

    # Predict Future AQI
    if model is not None:
        # Prepare feature vector for prediction matching exact training columns
        df_feat = df_live.copy()
        df_feat["hour"] = df_feat["date_time"].dt.hour
        df_feat["day_of_week"] = df_feat["date_time"].dt.dayofweek
        df_feat["month"] = df_feat["date_time"].dt.month
        df_feat["aqi_lag_1h"] = df_feat["us_aqi"].shift(1).fillna(current_aqi)
        df_feat["aqi_lag_3h"] = df_feat["us_aqi"].shift(3).fillna(current_aqi)
        df_feat["aqi_change_rate"] = df_feat["us_aqi"] - df_feat["aqi_lag_1h"]

        feature_cols = [
            'pm10', 'pm2_5', 'co', 'no2', 'so2', 'ozone', 'us_aqi',
            'hour', 'day_of_week', 'month', 'aqi_lag_1h', 'aqi_lag_3h', 'aqi_change_rate'
        ]

        predicted_aqi = int(model.predict(df_feat[feature_cols].iloc[[0]])[0])
        pred_status, pred_emoji, _ = get_aqi_status(predicted_aqi)

        st.subheader("🤖 Model Forecast (Next 24 Hours)")
        st.info(f"Forecasted AQI for tomorrow: **{predicted_aqi} ({pred_status} {pred_emoji})**")

        # Alert Banner for High AQI
        if predicted_aqi > 150:
            st.error(f"🚨 **HAZARDOUS AIR QUALITY ALERT:** The forecasted AQI is **{predicted_aqi}**. Sensitive individuals and outdoor activities should be limited!")

    # Interactive Trend Plot
    st.subheader("📈 3-Day Air Quality Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_live['date_time'], 
        y=df_live['us_aqi'],
        mode='lines+markers',
        name='Hourly AQI',
        line=dict(color='#1f77b4', width=3)
    ))
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="US AQI Index",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature Importance Section
    if os.path.exists("model_registry/shap_summary.png"):
        st.divider()
        st.subheader("💡 Model Explainability (SHAP Importance)")
        st.image("model_registry/shap_summary.png", caption="Key features influencing AQI predictions", use_column_width=True)

except Exception as e:
    st.error(f"Error loading dashboard data: {e}")
