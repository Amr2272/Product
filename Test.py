import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime, timedelta
import pickle
import os
import json

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(layout="wide", page_title="Sales Forecasting App")

# ---------------------------
# Paths
# ---------------------------
DATA_FILEPATH = "sales_data.csv"
MODEL_PATH = "prophet_model.pkl"

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['ds'])
    return df

df = load_data(DATA_FILEPATH)

# ---------------------------
# Load Model
# ---------------------------
def load_model(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        return None

model = load_model(MODEL_PATH)

# ---------------------------
# Real-time Alarm Function
# ---------------------------
def check_alarm(value, threshold):
    """
    Returns True if the value exceeds threshold.
    Can be used for real-time monitoring.
    """
    return value > threshold

# Example threshold
ALARM_THRESHOLD = 1000

# ---------------------------
# Sidebar Options
# ---------------------------
mode = st.sidebar.radio("Select Mode", ["Batch Forecast", "Real-time Monitoring"])

# ---------------------------
# Batch Forecast
# ---------------------------
if mode == "Batch Forecast":
    st.header("Batch Forecast")
    future_days = st.number_input("Days to forecast", min_value=1, max_value=365, value=30)

    if model:
        future = model.make_future_dataframe(periods=future_days)
        forecast = model.predict(future)

        # Show forecast plot
        fig = px.line(forecast, x='ds', y='yhat', title="Forecasted Sales")
        st.plotly_chart(fig, use_container_width=True)

        # Batch Alarm Example
        max_forecast = forecast['yhat'].max()
        if check_alarm(max_forecast, ALARM_THRESHOLD):
            st.error(f"⚠️ Batch Alarm Triggered! Max forecast value {max_forecast:.2f} exceeds threshold {ALARM_THRESHOLD}")

# ---------------------------
# Real-time Monitoring
# ---------------------------
elif mode == "Real-time Monitoring":
    st.header("Real-time Monitoring")
    
    # Simulate streaming data
    new_value = st.number_input("Enter new sales value", value=0)

    # Real-time Alarm
    if check_alarm(new_value, ALARM_THRESHOLD):
        st.error(f"⚠️ Real-time Alarm Triggered! Value {new_value:.2f} exceeds threshold {ALARM_THRESHOLD}")

    st.write("Current Value:", new_value)

# ---------------------------
# View API Response (Optional)
# ---------------------------
if st.checkbox("View API Response"):
    st.json({
        "model_loaded": model is not None,
        "data_rows": len(df),
        "mode_selected": mode
    })
