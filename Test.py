import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from prophet import Prophet
from datetime import date, timedelta
import os

st.set_page_config(layout="wide", page_title="Sales Forecast App", page_icon="📊")
MODEL_PATH = "prophet.pkl"

@st.cache_data
def load_data():
    try:
        train = pd.read_csv("Data.zip")
    except FileNotFoundError:
        dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
        sales = np.random.randint(50, 500, size=len(dates))
        train = pd.DataFrame({'ds': dates, 'y': sales})
    train['ds'] = pd.to_datetime(train['ds'])
    return train

@st.cache_resource
def load_model(path):
    if path and os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def check_mae(actual_df, forecast_df, threshold=0.25):
    df = actual_df.merge(forecast_df[['ds', 'yhat']], on='ds', how='inner')
    if df.empty:
        return None, None, "Error"
    df['abs_error'] = abs(df['y'] - df['yhat'])
    mae = df['abs_error'].mean()
    thresh_val = df['y'].mean() * threshold
    alert = "ALERT: High MAE" if mae > thresh_val else "Performance OK"
    return mae, thresh_val, alert

def run_dashboard(data):
    st.title("🛍️ City Sales Dashboard")
    total_records = data.shape[0]
    st.metric("Total Records", total_records)
    fig = px.line(data, x='ds', y='y', title="Sales Over Time")
    st.plotly_chart(fig, use_container_width=True)

def run_forecast(data, model):
    st.title("📈 Time Series Forecast")
    last_date = data['ds'].max()
    forecast_days = st.number_input("Days to Forecast", min_value=1, max_value=365, value=30)
    if st.button("Run Forecast"):
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        st.subheader("Forecast Results")
        st.dataframe(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(forecast_days))
        mae, thresh_val, alert = check_mae(data, forecast)
        if mae:
            if "ALERT" in alert:
                st.error(f"{alert}: MAE={mae:.2f}, Threshold={thresh_val:.2f}")
            else:
                st.success(f"{alert}: MAE={mae:.2f}, Threshold={thresh_val:.2f}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines+markers', name='Actual'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
            y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
            fill='toself', fillcolor='rgba(27,188,156,0.2)', line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval', hoverinfo="skip"
        ))
        fig.update_layout(title="Forecast vs Actual", xaxis_title="Date", yaxis_title="Sales")
        st.plotly_chart(fig, use_container_width=True)

data = load_data()
model = load_model(MODEL_PATH)

if model is None:
    model = Prophet()
    model.fit(data)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

st.sidebar.title("App Navigation")
mode = st.sidebar.selectbox("Choose Mode", ["Dashboard", "Forecast"])
if mode == "Dashboard":
    run_dashboard(data)
else:
    run_forecast(data, model)
