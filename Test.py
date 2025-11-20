import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from prophet import Prophet
from datetime import date, timedelta, datetime
import os
import plotly.graph_objects as go
from typing import Dict, List, Optional
import json

st.set_page_config(layout="wide", page_title="Data Analysis & Forecast App", page_icon="📊")
MODEL_PATH = 'prophet.pkl'

# ============================================================================
# PREDICTION MODE CONFIGURATION
# ============================================================================

class PredictionMode:
    """Configuration for different prediction modes"""
    BATCH = "batch"
    REAL_TIME = "real_time"
    SCHEDULED = "scheduled"

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_data():
    try:
        train = pd.read_csv('Data.zip')
    except FileNotFoundError:
       return('Error When Loading Data')
    
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

# ============================================================================
# BATCH PREDICTION FUNCTIONS
# ============================================================================

def batch_predict(model, periods: int, include_history: bool = True) -> pd.DataFrame:
    """
    Generate batch predictions for multiple future periods
    
    Args:
        model: Trained Prophet model
        periods: Number of days to forecast
        include_history: Whether to include historical fitted values
    
    Returns:
        DataFrame with predictions
    """
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    if not include_history:
        # Filter to only future dates
        last_train_date = model.history['ds'].max()
        forecast = forecast[forecast['ds'] > last_train_date]
    
    return forecast

# ============================================================================
# REAL-TIME PREDICTION FUNCTIONS
# ============================================================================

def real_time_predict(model, target_date: datetime, 
                      context_data: Optional[Dict] = None) -> Dict:
    """
    Generate real-time prediction for a specific date with optional context
    
    Args:
        model: Trained Prophet model
        target_date: Specific date to predict
        context_data: Optional dictionary with additional context (promotions, holidays, etc.)
    
    Returns:
        Dictionary with prediction results
    """
    # Create a future dataframe for the single target date
    future_df = pd.DataFrame({'ds': [pd.to_datetime(target_date)]})
    
    # Add regressor values if context data provided
    if context_data:
        for key, value in context_data.items():
            if key in model.extra_regressors:
                future_df[key] = value
    
    # Generate prediction
    forecast = model.predict(future_df)
    
    # Extract key metrics
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

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

def check_model_performance(prophet_df, forecast_data, mae_percent_threshold=0.25):
    """
    Calculates MAE on historical data fit and checks against a percentage threshold
    """
    performance_df = prophet_df.merge(
        forecast_data[['ds', 'yhat']], 
        on='ds', 
        how='inner'
    )
    
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

# ============================================================================
# DASHBOARD FUNCTION
# ============================================================================

def run_dashboard(train, min_date, max_date, sort_state):
    st.title("🛒️ Store Sales Forecasting Project")
    st.markdown("""
        ### 📋 Project Overview
        This project aims to forecast store sales using historical data.
        Throughout the process, several data issues were identified and resolved to improve model performance.
        
        ### ⚙️ Data Cleaning & Preprocessing
        - **Handled Missing Values:** Filled or removed missing entries to maintain data consistency.  
        - **Removed Duplicates:** Ensured no duplicate records exist in the dataset.  
        - **Corrected Holiday Data:** Removed or adjusted incorrect duplicate holiday entries.
        """)
    
    st.title("City Sales Dashboard")

    st.subheader("Data Summary (Unique Values)")
    
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

    with st.expander("View Unique Lists for Categories"):
        for col in ['state', 'city', 'store_nbr', 'family', 'store_type', 'cluster', 'day_type']:
            if col in train.columns:
                st.write(f"**Unique {col.capitalize()}s:**")
                st.code(', '.join(sorted(train[col].unique().astype(str))))
        
    st.markdown("---")

    st.subheader("Dashboard Controls")

    col_chosen = st.multiselect(
        "Choose State",
        options=sort_state.index.tolist(),
        default=['Pichincha'] if 'Pichincha' in sort_state.index else (sort_state.index.tolist()[:1] if not sort_state.empty else []),
        key='state_id_dash',
        placeholder="Select one or more states"
    )

    agg_method = st.radio(
        "Aggregate Method",
        options=['sum', 'mean'],
        index=0,  
        horizontal=True,
        key='value_id_dash'
    )

    st.markdown("---")
    st.subheader("Date Range Selection")

    date_range = st.date_input(
        "Select Date Range",
        value=[min_date, max_date],
        min_value=date(2000, 1, 1), 
        max_value=date(2100, 1, 1), 
        key='date_id_dash'
    )

    if len(date_range) == 2:
        start_date, end_date = date_range[0], date_range[1]
    elif len(date_range) == 1:
        start_date, end_date = date_range[0], date_range[0]
    else:
        start_date, end_date = min_date, max_date

    st.caption(f"Selected range: **{start_date.strftime('%d-%m-%Y')}** → **{end_date.strftime('%d-%m-%Y')}**")

    if 'sales' not in train.columns:
        st.error("The 'sales' column is missing from the data. Cannot generate sales-related charts.")
        return

    if not col_chosen:
        st.warning("Please select at least one state to view sales data.")
        return

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    try:
        mask = (
            (train['state'].isin(col_chosen)) &
            (train.index >= start_date) &
            (train.index <= end_date)
        )
        filterr = train[mask]

        if filterr.empty:
            st.warning("No data found for the selected date range and filters.")
            fig = px.bar(title="No Data for Selected Filters")

        else:
            if agg_method == 'mean':
                city_sales = filterr.groupby('city')['sales'].mean().reset_index()
            else:
                city_sales = filterr.groupby('city')['sales'].sum().reset_index()

            city_sales['sales'] = pd.to_numeric(city_sales['sales'], errors='coerce').fillna(0)

            fig = px.bar(
                city_sales, x='city', y='sales',
                labels={'city': 'City', 'sales': f'Sales ({agg_method.capitalize()})'},
                title=f'City Sales for Selected States ({agg_method.capitalize()})',
                text=city_sales['sales'].apply(lambda x: f'{x:,.0f}'), 
                color_discrete_sequence=["#1abc9c"]
            )

            fig.update_traces(textposition='outside')
            fig.update_yaxes(tickformat=".2s", title_font=dict(size=14))
            fig.update_xaxes(title_font=dict(size=14))
            fig.update_layout(title_font_size=20)

        st.plotly_chart(fig, width='stretch')

    except Exception as e:
        st.error(f"An error occurred while processing data: {str(e)}")

# ============================================================================
# ENHANCED FORECAST APP WITH BATCH & REAL-TIME MODES
# ============================================================================

def run_forecast_app(model, prophet_df):
    st.title("📈 Time Series Forecasting (Prophet)")

    if model is None or prophet_df.empty:
        st.error("Prophet model or historical data is missing. Cannot run forecast.")
        return

    # ========================================================================
    # PREDICTION MODE SELECTOR
    # ========================================================================
    
    st.sidebar.header("🎯 Prediction Mode")
    
    prediction_mode = st.sidebar.radio(
        "Select Prediction Type",
        options=["📦 Batch Predictions", "⚡ Real-Time Predictions"],
        key='prediction_mode'
    )
    
    st.sidebar.markdown("---")
    
    # ========================================================================
    # MODE 1: BATCH PREDICTIONS
    # ========================================================================
    
    if prediction_mode == "📦 Batch Predictions":
        st.header("📦 Batch Prediction Mode")
        st.info("Generate forecasts for multiple days at once. Ideal for planning and reporting.")
        
        st.sidebar.header("Batch Forecast Settings")
        
        last_train_date = prophet_df['ds'].max()
        st.sidebar.info(f"Last historical date: **{last_train_date.strftime('%Y-%m-%d')}**")

        forecast_end_date = st.sidebar.date_input(
            "Select Forecast End Date:",
            value=last_train_date.date() + timedelta(days=30),
            min_value=date(2000, 1, 1), 
            max_value=date(2100, 1, 1), 
            key='date_id_forecast'
        )
        
        periods = (pd.to_datetime(forecast_end_date) - last_train_date).days
        
        if periods > 0:
            st.sidebar.success(f"Forecasting **{periods}** days.")
        else:
            st.sidebar.warning("Selected date is not in the future.")
        
        if st.sidebar.button("🚀 Run Batch Forecast", key='batch_forecast_button'):
            if periods > 0:
                with st.spinner('Generating batch forecast...'):
                    forecast = batch_predict(model, periods, include_history=True)
                    
                    last_train_date_only = last_train_date.date()
                    forecast_future = forecast[forecast['ds'].dt.date > last_train_date_only].copy()
                    
                    st.session_state.forecast_data = forecast
                    st.session_state.forecast_future_data = forecast_future
                    st.session_state.model_fit = model
                    
                    # Performance check
                    MAE_PERCENT_THRESHOLD = 0.25
                    mae_result, threshold_value, alert_status = check_model_performance(
                        prophet_df, forecast, MAE_PERCENT_THRESHOLD
                    )
                    
                    st.session_state.mae = mae_result
                    st.session_state.mae_threshold_value = threshold_value
                    st.session_state.alert_status = alert_status
                    st.session_state.mae_percent_threshold = MAE_PERCENT_THRESHOLD

                st.success(f"✅ Batch forecast generated for {periods} days!")
            else:
                st.error("Forecast end date must be after the last training date.")
        
        # Display batch results
        if st.session_state.forecast_data is not None:
            display_forecast_results(
                st.session_state.forecast_data,
                st.session_state.forecast_future_data,
                st.session_state.model_fit,
                prophet_df
            )
    
    # ========================================================================
    # MODE 2: REAL-TIME PREDICTIONS
    # ========================================================================
    
    elif prediction_mode == "⚡ Real-Time Predictions":
        st.header("⚡ Real-Time Prediction Mode")
        st.info("Generate instant predictions for specific dates. Ideal for API integration and on-demand forecasting.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            target_date = st.date_input(
                "Select Target Date for Prediction:",
                value=datetime.now().date() + timedelta(days=1),
                min_value=date(2000, 1, 1),
                max_value=date(2100, 1, 1),
                key='real_time_date'
            )
        
        with col2:
            st.markdown("###")
            if st.button("🎯 Predict Now", key='real_time_button', type="primary"):
                with st.spinner('Generating real-time prediction...'):
                    result = real_time_predict(model, pd.to_datetime(target_date))
                    
                    # Store in session state
                    st.session_state.real_time_predictions.insert(0, result)
                    if len(st.session_state.real_time_predictions) > 10:
                        st.session_state.real_time_predictions.pop()
                
                st.success("✅ Prediction complete!")
        
        # Display real-time result
        if st.session_state.real_time_predictions:
            latest = st.session_state.real_time_predictions[0]
            
            st.markdown("### 📊 Latest Prediction")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Date", latest['date'].strftime('%Y-%m-%d'))
            with col2:
                st.metric("Prediction", f"{latest['prediction']:,.0f}")
            with col3:
                st.metric("Lower Bound", f"{latest['lower_bound']:,.0f}")
            with col4:
                st.metric("Upper Bound", f"{latest['upper_bound']:,.0f}")
            
            # Show JSON response (API simulation)
            with st.expander("🔍 View API Response (JSON)"):
                st.json(latest)
            
            # History of predictions
            if len(st.session_state.real_time_predictions) > 1:
                st.markdown("### 📜 Prediction History")
                history_df = pd.DataFrame(st.session_state.real_time_predictions)
                history_df['date'] = pd.to_datetime(history_df['date']).dt.strftime('%Y-%m-%d')
                st.dataframe(
                    history_df[['date', 'prediction', 'lower_bound', 'upper_bound']],
                    width='stretch'
                )


def display_forecast_results(forecast_data, forecast_future, model_fit, prophet_df):
    """Display forecast results with performance metrics"""
    
    # Performance metrics
    if 'mae' in st.session_state and st.session_state.mae is not None:
        st.markdown("---")
        st.subheader("Model Performance & Alert System 🔔")

        mae_val = st.session_state.mae
        alert_status = st.session_state.alert_status
        threshold_value = st.session_state.mae_threshold_value
        percent_threshold = st.session_state.mae_percent_threshold * 100
        
        if "ALERT" in alert_status:
            st.error(f"**🚨 {alert_status}**\n\n**Action Required:** Prediction accuracy has dropped below the defined threshold.\n\n*Threshold: < {percent_threshold:.0f}% of Mean Sales (Calculated Value: {threshold_value:,.2f} MAE)*")
        else:
            st.info(f"**✅ {alert_status}**\n\nPrediction accuracy is within the acceptable threshold.")
        
        st.markdown(f"""
        > **Performance Log**
        > 
        > * **Metric Used:** Mean Absolute Error (MAE) on historical fit.
        > * **Calculated MAE:** **{mae_val:,.2f}**
        > * **Defined Threshold:** **{percent_threshold:.0f}% of Mean Sales (Actual Value: {threshold_value:,.2f} MAE)**
        > * **Run Date:** {pd.Timestamp('now').strftime('%Y-%m-%d %H:%M:%S')}
        """)
    
    st.markdown("---")
    st.header("Forecast Results")
    
    # Forecast table
    st.subheader("Forecast Table (Future Days)")
    if not forecast_future.empty:
        st.dataframe(
            forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
                columns={'ds': 'Date', 'yhat': 'Forecast Value', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'}
            ).set_index('Date').style.format({'Forecast Value': "{:,.0f}", 'Lower Bound': "{:,.0f}", 'Upper Bound': "{:,.0f}"}),
            width='stretch'
        )
    else:
        st.warning("No future dates found in the forecast data.")

    # Visualization
    st.subheader("Forecast Visualization")
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=prophet_df['ds'],
        y=prophet_df['y'],
        mode='markers',
        name='Historical Data (Actual)',
        marker=dict(color='blue', size=4)
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat'],
        mode='lines',
        name='Forecast (Predicted)',
        line=dict(color='#1abc9c', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_data['ds'], forecast_data['ds'].iloc[::-1]]),
        y=pd.concat([forecast_data['yhat_upper'], forecast_data['yhat_lower'].iloc[::-1]]),
        fill='toself',
        fillcolor='rgba(27, 188, 156, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='80% Confidence Interval'
    ))

    fig.update_layout(
        title='Historical Data and Future Forecast',
        xaxis_title='Date',
        yaxis_title='Value',
        title_font_size=20
    )
    
    st.plotly_chart(fig, width='stretch')

    # Components
    st.subheader("Model Components")
    fig_components = model_fit.plot_components(forecast_data)
    st.write(fig_components)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

if __name__ == '__main__':
    # Initialize ALL session state at the very beginning
    if 'forecast_data' not in st.session_state:
        st.session_state.forecast_data = None
    if 'forecast_future_data' not in st.session_state:
        st.session_state.forecast_future_data = None
    if 'model_fit' not in st.session_state:
        st.session_state.model_fit = None
    if 'mae' not in st.session_state:
        st.session_state.mae = None
    if 'mae_threshold_value' not in st.session_state:
        st.session_state.mae_threshold_value = None
    if 'alert_status' not in st.session_state:
        st.session_state.alert_status = None
    if 'mae_percent_threshold' not in st.session_state:
        st.session_state.mae_percent_threshold = None
    if 'real_time_predictions' not in st.session_state:
        st.session_state.real_time_predictions = []
    
    train, min_date, max_date, sort_state, prophet_df = load_data()
    model = load_prophet_model(MODEL_PATH)
    
    st.sidebar.title("App Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose App Mode",
        ["City Sales Dashboard", "Time Series Forecast"]
    )

    if app_mode == "City Sales Dashboard":
        if st.session_state.forecast_data is not None:
            st.session_state.forecast_data = None
            st.session_state.mae = None
            st.session_state.alert_status = None
        run_dashboard(train, min_date, max_date, sort_state)
    elif app_mode == "Time Series Forecast":
        run_forecast_app(model, prophet_df)




