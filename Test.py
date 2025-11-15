import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from prophet import Prophet
from datetime import date, timedelta, datetime
import os
import plotly.graph_objects as go
from typing import Dict, Optional

st.set_page_config(layout="wide", page_title="Data Analysis & Forecast App", page_icon="📊")
MODEL_PATH = 'prophet.pkl'

class PredictionMode:
    BATCH = "batch"
    REAL_TIME = "real_time"
    SCHEDULED = "scheduled"

@st.cache_data
def load_data():
    try:
        train = pd.read_csv('Data.zip')
    except FileNotFoundError:
        dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
        states = ['Pichincha', 'Guayas', 'Azuay', 'Manabi', 'El Oro']
        store_types = ['A', 'B', 'C', 'D']
        families = ['AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS', 'BREAD/BAKERY', 'CLEANING', 'DAIRY']
        mock_data = []
        record_id = 0
        for day in dates:
            dcoilwtico = round(np.random.uniform(30, 100), 2)
            for store_nbr in range(1, 6):
                state = np.random.choice(states)
                if state == 'Pichincha':
                    city = np.random.choice(['Quito', 'Rumiñahui'])
                elif state == 'Guayas':
                    city = np.random.choice(['Guayaquil', 'Daule'])
                elif state == 'Azuay':
                    city = np.random.choice(['Cuenca'])
                else:
                    city = np.random.choice(['City X', 'City Y'])
                store_type = np.random.choice(store_types)
                cluster = np.random.randint(1, 18)
                for family in families:
                    sales = np.random.randint(0, 500) if np.random.rand() > 0.1 else 0
                    onpromotion = np.random.randint(0, 50) if sales > 0 else 0
                    day_type = np.random.choice(['Holiday', 'Work Day', 'Weekend'])
                    mock_data.append([day, record_id, store_nbr, family, sales, onpromotion, city, state, store_type, cluster, dcoilwtico, day_type])
                    record_id += 1
        train = pd.DataFrame(mock_data, columns=['date', 'id', 'store_nbr', 'family', 'sales', 'onpromotion', 'city', 'state', 'store_type', 'cluster', 'dcoilwtico', 'day_type'])
    train["date"] = pd.to_datetime(train["date"], errors="coerce")
    train = train.dropna(subset=['date'])
    train = train.set_index("date")
    train.index = pd.to_datetime(train.index)
    min_date = train.index.min().date()
    max_date = train.index.max().date()
    sort_state = train.groupby('state')['sales'].sum().sort_values(ascending=False) if 'state' in train.columns and 'sales' in train.columns else pd.Series([], dtype='float64')
    prophet_df = train.groupby(train.index)['sales'].sum().reset_index()
    prophet_df.columns = ['ds', 'y']
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    return train, min_date, max_date, sort_state, prophet_df

@st.cache_resource
def load_prophet_model(path):
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception:
            return None
    else:
        return None

def batch_predict(model, periods: int, include_history: bool = True) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    if not include_history:
        last_train_date = model.history['ds'].max()
        forecast = forecast[forecast['ds'] > last_train_date]
    return forecast

def real_time_predict(model, target_date: datetime, context_data: Optional[Dict] = None) -> Dict:
    future_df = pd.DataFrame({'ds': [pd.to_datetime(target_date)]})
    if context_data:
        for key, value in context_data.items():
            if key in model.extra_regressors:
                future_df[key] = value
    forecast = model.predict(future_df)
    result = {
        'date': target_date,
        'prediction': float(forecast['yhat'].iloc[0]),
        'lower_bound': float(forecast['yhat_lower'].iloc[0]),
        'upper_bound': float(forecast['yhat_upper'].iloc[0]),
        'trend': float(forecast['trend'].iloc[0]),
        'timestamp': datetime.now().isoformat(),
        'context': context_data or {}
    }
    return result

def check_model_performance(prophet_df, forecast_data, mae_percent_threshold=0.25):
    performance_df = prophet_df.merge(forecast_data[['ds', 'yhat']], on='ds', how='inner')
    if not performance_df.empty:
        performance_df['abs_error'] = abs(performance_df['y'] - performance_df['yhat'])
        mae = performance_df['abs_error'].mean()
        mean_y = performance_df['y'].mean()
        threshold_value = mean_y * mae_percent_threshold
        if mae > threshold_value:
            alert_status = "ALERT: High MAE"
        else:
            alert_status = "Performance OK"
        return mae, threshold_value, alert_status
    return None, None, "Error: Historical data missing or merge failed."

def run_dashboard(train, min_date, max_date, sort_state):
    st.title("🛒️ Store Sales Forecasting Project")
    st.title("City Sales Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    total_records = train.shape[0]
    unique_states = train['state'].nunique() if 'state' in train.columns else 0
    unique_cities = train['city'].nunique() if 'city' in train.columns else 0
    unique_stores = train['store_nbr'].nunique() if 'store_nbr' in train.columns else 0
    col1.metric("Total Records", f"{total_records:,}")
    col2.metric("Unique States", unique_states)
    col3.metric("Unique Cities", unique_cities)
    col4.metric("Unique Stores", unique_stores)
    col5, col6, col7 = st.columns(3)
    unique_families = train['family'].nunique() if 'family' in train.columns else 0
    unique_store_types = train['store_type'].nunique() if 'store_type' in train.columns else 0
    unique_clusters = train['cluster'].nunique() if 'cluster' in train.columns else 0
    col5.metric("Unique Families", unique_families)
    col6.metric("Unique Store Types", unique_store_types)
    col7.metric("Unique Clusters", unique_clusters)

def run_forecast_app(model, prophet_df):
    st.title("📈 Time Series Forecasting (Prophet)")
    if model is None or prophet_df.empty:
        st.error("Prophet model or historical data is missing. Cannot run forecast.")
        return
    st.sidebar.header("🎯 Prediction Mode")
    prediction_mode = st.sidebar.radio("Select Prediction Type", options=["📦 Batch Predictions", "⚡ Real-Time Predictions"], key='prediction_mode')
    if prediction_mode == "📦 Batch Predictions":
        st.header("📦 Batch Prediction Mode")
        last_train_date = prophet_df['ds'].max()
        forecast_end_date = st.sidebar.date_input("Select Forecast End Date:", value=last_train_date.date() + timedelta(days=30), min_value=date(2000, 1, 1), max_value=date(2100, 1, 1), key='date_id_forecast')
        periods = (pd.to_datetime(forecast_end_date) - last_train_date).days
        if st.sidebar.button("🚀 Run Batch Forecast", key='batch_forecast_button'):
            if periods > 0:
                forecast = batch_predict(model, periods, include_history=True)
                st.session_state.forecast_data = forecast
                st.session_state.forecast_future_data = forecast[forecast['ds'].dt.date > last_train_date.date()]
                mae_result, threshold_value, alert_status = check_model_performance(prophet_df, forecast)
                st.session_state.mae = mae_result
                st.session_state.mae_threshold_value = threshold_value
                st.session_state.alert_status = alert_status
    elif prediction_mode == "⚡ Real-Time Predictions":
        st.header("⚡ Real-Time Prediction Mode")
        target_date = st.date_input("Select Target Date for Prediction:", value=datetime.now().date() + timedelta(days=1), min_value=date(2000, 1, 1), max_value=date(2100, 1, 1), key='real_time_date')
        if st.button("🎯 Predict Now", key='real_time_button', type="primary"):
            result = real_time_predict(model, pd.to_datetime(target_date))
            REAL_TIME_THRESHOLD = 1000
            if result['prediction'] > REAL_TIME_THRESHOLD:
                st.error(f"⚠️ Real-Time Alarm! Prediction {result['prediction']:.0f} exceeds threshold {REAL_TIME_THRESHOLD}")
            st.session_state.real_time_predictions.insert(0, result)
            if len(st.session_state.real_time_predictions) > 10:
                st.session_state.real_time_predictions.pop()
        if st.session_state.real_time_predictions:
            latest = st.session_state.real_time_predictions[0]
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Date", latest['date'].strftime('%Y-%m-%d'))
            with col2: st.metric("Prediction", f"{latest['prediction']:,.0f}")
            with col3: st.metric("Lower Bound", f"{latest['lower_bound']:,.0f}")
            with col4: st.metric("Upper Bound", f"{latest['upper_bound']:,.0f}")
            with st.expander("🔍 View API Response (JSON)"):
                st.json(latest)

if __name__ == '__main__':
    if 'forecast_data' not in st.session_state: st.session_state.forecast_data = None
    if 'forecast_future_data' not in st.session_state: st.session_state.forecast_future_data = None
    if 'model_fit' not in st.session_state: st.session_state.model_fit = None
    if 'mae' not in st.session_state: st.session_state.mae = None
    if 'mae_threshold_value' not in st.session_state: st.session_state.mae_threshold_value = None
    if 'alert_status' not in st.session_state: st.session_state.alert_status = None
    if 'mae_percent_threshold' not in st.session_state: st.session_state.mae_percent_threshold = None
    if 'real_time_predictions' not in st.session_state: st.session_state.real_time_predictions = []
    train, min_date, max_date, sort_state, prophet_df = load_data()
    model = load_prophet_model(MODEL_PATH)
    st.sidebar.title("App Navigation")
    app_mode = st.sidebar.selectbox("Choose App Mode", ["City Sales Dashboard", "Time Series Forecast"])
    if app_mode == "City Sales Dashboard":
        run_dashboard(train, min_date, max_date, sort_state)
    elif app_mode == "Time Series Forecast":
        run_forecast_app(model, prophet_df)
