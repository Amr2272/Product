import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from prophet import Prophet
from datetime import date, timedelta
import os
import plotly.graph_objects as go

# -----------------------------------------------------------
# ---------------  STREAMLIT CONFIG  ------------------------
# -----------------------------------------------------------

st.set_page_config(
    layout="wide",
    page_title="Data Analysis & Forecast App",
    page_icon="📊"
)

MODEL_PATH = "prophet.pkl"

# -----------------------------------------------------------
# --------------------- DATA LOADER --------------------------
# -----------------------------------------------------------

@st.cache_data
def load_data():
    try:
        train = pd.read_csv("Data.zip")
    except FileNotFoundError:
        # Create MOCK DATA (same as your original code)
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")

        states = ["Pichincha", "Guayas", "Azuay", "Manabi", "El Oro"]
        store_types = ["A", "B", "C", "D"]
        families = ["AUTOMOTIVE", "BABY CARE", "BEAUTY", "BEVERAGES",
                    "BOOKS", "BREAD/BAKERY", "CLEANING", "DAIRY"]

        mock_data = []
        record_id = 0
        for day in dates:
            dcoil = round(np.random.uniform(30, 100), 2)

            for store_nbr in range(1, 6):
                state = np.random.choice(states)

                if state == "Pichincha":
                    city = np.random.choice(["Quito", "Rumiñahui"])
                elif state == "Guayas":
                    city = np.random.choice(["Guayaquil", "Daule"])
                elif state == "Azuay":
                    city = "Cuenca"
                else:
                    city = np.random.choice(["City X", "City Y"])

                store_type = np.random.choice(store_types)
                cluster = np.random.randint(1, 18)

                for family in families:
                    sales = np.random.randint(0, 500) if np.random.rand() > 0.1 else 0
                    promo = np.random.randint(0, 50) if sales > 0 else 0
                    day_type = np.random.choice(["Holiday", "Work Day", "Weekend"])

                    mock_data.append([
                        day, record_id, store_nbr, family, sales, promo,
                        city, state, store_type, cluster, dcoil, day_type
                    ])
                    record_id += 1

        train = pd.DataFrame(mock_data, columns=[
            "date", "id", "store_nbr", "family", "sales", "onpromotion",
            "city", "state", "store_type", "cluster", "dcoilwtico", "day_type"
        ])

    train["date"] = pd.to_datetime(train["date"], errors="coerce")
    train = train.dropna(subset=["date"])
    train = train.set_index("date")

    min_date = train.index.min().date()
    max_date = train.index.max().date()

    sort_state = train.groupby("state")["sales"].sum().sort_values(ascending=False)

    prophet_df = train.groupby(train.index)["sales"].sum().reset_index()
    prophet_df.columns = ["ds", "y"]

    return train, min_date, max_date, sort_state, prophet_df


# -----------------------------------------------------------
# ------------------ LOAD PROPHET MODEL ----------------------
# -----------------------------------------------------------

@st.cache_resource
def load_prophet_model(path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


# -----------------------------------------------------------
# -------------- NEW: REAL-TIME PREDICTION ------------------
# -----------------------------------------------------------

def predict_single(model, date_input):
    """Single real-time date prediction."""
    future = pd.DataFrame({"ds": [pd.to_datetime(date_input)]})
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def predict_batch(model, dates_list):
    """Batch prediction for multiple dates."""
    future = pd.DataFrame({"ds": pd.to_datetime(dates_list)})
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


# -----------------------------------------------------------
# ----------------- DASHBOARD PAGE ---------------------------
# -----------------------------------------------------------

def run_dashboard(train, min_date, max_date, sort_state):
    st.title("🛍️ Store Sales Dashboard")

    st.subheader("Data Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{train.shape[0]:,}")
    col2.metric("States", train["state"].nunique())
    col3.metric("Cities", train["city"].nunique())
    col4.metric("Stores", train["store_nbr"].nunique())

    st.markdown("---")

    state_select = st.multiselect(
        "Select State(s)",
        options=sort_state.index.tolist(),
        default=sort_state.index[:1]
    )

    date_range = st.date_input(
        "Select Date Range",
        [min_date, max_date]
    )

    if len(date_range) == 2:
        start_date, end_date = date_range

    mask = (
        (train["state"].isin(state_select)) &
        (train.index >= pd.to_datetime(start_date)) &
        (train.index <= pd.to_datetime(end_date))
    )

    filtered = train[mask]

    if filtered.empty:
        st.warning("No data for selected filters.")
        return

    city_sales = filtered.groupby("city")["sales"].sum().reset_index()

    fig = px.bar(
        city_sales,
        x="city",
        y="sales",
        title="City Sales",
        text="sales"
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------
# ----------------- FORECAST PAGE ----------------------------
# -----------------------------------------------------------

def run_forecast_app(model, prophet_df):
    st.title("📈 Time Series Forecast (Prophet)")

    last_date = prophet_df["ds"].max()

    st.sidebar.header("Forecast Settings")

    forecast_end = st.sidebar.date_input(
        "Forecast End Date",
        value=last_date.date() + timedelta(days=30)
    )

    periods = (pd.to_datetime(forecast_end) - last_date).days

    if st.sidebar.button("Run Forecast"):
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        st.subheader("Forecast Table")
        st.dataframe(
            forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=prophet_df["ds"],
            y=prophet_df["y"],
            mode="markers",
            name="Actual"
        ))

        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            mode="lines",
            name="Forecast"
        ))

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ----------------------- REAL-TIME MODE ----------------------------
    st.subheader("🔮 Real-Time Prediction")

    rt_date = st.date_input("Select a date for real-time forecast")

    if st.button("Predict Real-Time"):
        result = predict_single(model, rt_date)
        st.write(result)

    # ----------------------- BATCH MODE ----------------------------
    st.subheader("📦 Batch Prediction")

    batch_dates = st.date_input("Select multiple dates", [])

    if st.button("Predict Batch"):
        if batch_dates:
            result = predict_batch(model, batch_dates)
            st.write(result)
        else:
            st.warning("Please select dates first.")


# -----------------------------------------------------------
# ------------------------ MAIN APP -------------------------
# -----------------------------------------------------------

if __name__ == "__main__":
    train, min_date, max_date, sort_state, prophet_df = load_data()
    model = load_prophet_model(MODEL_PATH)

    st.sidebar.title("Navigation")

    choice = st.sidebar.selectbox(
        "Choose Page",
        ["Dashboard", "Forecast"]
    )

    if choice == "Dashboard":
        run_dashboard(train, min_date, max_date, sort_state)
    else:
        run_forecast_app(model, prophet_df)
