# 🌫️ End-to-End Serverless AQI Predictor & MLOps Pipeline

An automated, serverless machine learning engineering pipeline that predicts the Air Quality Index (AQI) 24 hours into the future for Islamabad using real-time air quality metrics, feature engineering, an embedded feature store, and an interactive Streamlit web dashboard.

---

## 🚀 Key Features

* **Real-time Data Ingestion:** Fetches atmospheric pollutant data ($\text{PM}_{2.5}$, $\text{PM}_{10}$, $\text{NO}_2$, $\text{SO}_2$, $\text{CO}$, $\text{O}_3$) and US AQI standard metrics from the open-source **Open-Meteo API**.
* **Feature Engineering & Historical Backfill:** Computes temporal features (hour, day of week, month), lag dynamics (1-hour, 3-hour lag), and AQI change rates across 1 year of historical backfill data.
* **Serverless Feature Store:** Uses **DuckDB** as a lightweight analytical feature store for feature storage and retrieval.
* **Machine Learning Pipeline:** Evaluates linear and gradient-boosted regression models (Ridge Regression, LightGBM) using time-series split cross-validation.
* **Explainable AI (XAI):** Provides model explainability and feature importance analysis via **SHAP (SHapley Additive exPlanations)** values.
* **Interactive Web Dashboard:** Live web app built with **Streamlit** featuring real-time AQI tracking, 3-day trend visualizer, 24-hour predictions, and automated hazardous air quality alert banners.

---

## 🛠️ Tech Stack & Tools

| Component | Tool / Library | Description |
| :--- | :--- | :--- |
| **API / Data Source** | Open-Meteo Air Quality API | Free, open-source atmospheric and pollutant historical/forecast data |
| **Feature Store** | DuckDB | Embedded, serverless SQL analytical engine |
| **ML Frameworks** | Scikit-Learn, LightGBM | Ridge Regression baseline and LightGBM decision trees |
| **Model Explainability** | SHAP | Feature importance and tree/linear explanations |
| **Model Registry** | Local Artifacts (`joblib`) | Serialized binary storage for model checkpoints |
| **Web Dashboard** | Streamlit, Plotly | Interactive frontend dashboard and visualizer |

---

## 📊 Model Evaluation Results

Models were trained on **~8,700 hourly historical data points** with an 80/20 chronological time-series split:

| Model | MAE | RMSE | $R^2$ Score |
| :--- | :--- | :--- | :--- |
| **Ridge Regression (Deployed)** | **13.44** | **17.62** | **0.7136** |
| LightGBM Regressor | 14.66 | 19.03 | 0.6658 |

> **Key Insight:** The deployed Ridge Regression model captures **~71.4% of the variance** in 24-hour future AQI forecasts, with an average prediction error of only **~13 AQI points**.

---

