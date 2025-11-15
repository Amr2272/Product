import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from prophet import Prophet
from datetime import date, timedelta
import os
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(layout="wide", page_title="Data Analysis & Forecast App", page_icon="📊")
MODEL_PATH = 'prophet.pkl'

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

                    mock_data.append([
                        day, record_id, store_nbr, family, sales, onpromotion, 
                        city, state, store_type, cluster, dcoilwtico, day_type
                    ])
                    record_id += 1

        train = pd.DataFrame(mock_data, columns=[
            'date', 'id', 'store_nbr', 'family', 'sales', 'onpromotion', 
            'city', 'state', 'store_type', 'cluster', 'dcoilwtico', 'day_type'
        ])
    
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


def run_dashboard(train, min_date, max_date, sort_state):
    st.title("🛍️ Store Sales Forecasting Project")
    st.markdown("""
        ### 🧾 Project Overview
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

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while processing data: {str(e)}")


def check_model_performance(prophet_df, forecast_data, mae_percent_threshold=0.25):
    """
    Calculates MAE on historical data fit and checks against a percentage threshold
    of the mean actual sales (y).
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


def train_prophet_model(df, seasonality_mode='multiplicative'):
    """Train a Prophet model on the given data"""
    model = Prophet(seasonality_mode=seasonality_mode)
    model.fit(df)
    return model


def generate_batch_forecasts(train, forecast_end_date, batch_by='family', selected_items=None):
    """
    Generate batch forecasts for multiple segments
    
    Parameters:
    - train: Full training dataset
    - forecast_end_date: End date for forecasting
    - batch_by: Column to batch by ('family', 'store_nbr', 'state', 'store_type')
    - selected_items: List of items to forecast (if None, forecast all)
    
    Returns:
    - Dictionary of forecasts and performance metrics
    """
    results = {}
    
    if batch_by not in train.columns:
        return None
    
    # Get unique items to forecast
    items = selected_items if selected_items else train[batch_by].unique()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, item in enumerate(items):
        status_text.text(f"Processing {batch_by}: {item} ({idx+1}/{len(items)})")
        
        # Filter data for this item
        item_data = train[train[batch_by] == item].copy()
        
        # Prepare Prophet dataframe
        prophet_df = item_data.groupby(item_data.index)['sales'].sum().reset_index()
        prophet_df.columns = ['ds', 'y']
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        
        # Skip if insufficient data
        if len(prophet_df) < 30:
            results[item] = {'error': 'Insufficient data (< 30 days)'}
            continue
        
        try:
            # Train model
            model = train_prophet_model(prophet_df)
            
            # Generate forecast
            last_train_date = prophet_df['ds'].max()
            periods = (pd.to_datetime(forecast_end_date) - last_train_date).days
            
            if periods > 0:
                future = model.make_future_dataframe(periods=periods)
                forecast = model.predict(future)
                
                # Get future forecast only
                forecast_future = forecast[forecast['ds'].dt.date > last_train_date.date()].copy()
                
                # Check performance
                mae, threshold, alert = check_model_performance(prophet_df, forecast)
                
                results[item] = {
                    'forecast': forecast_future,
                    'full_forecast': forecast,
                    'historical': prophet_df,
                    'model': model,
                    'mae': mae,
                    'threshold': threshold,
                    'alert': alert,
                    'total_forecast': forecast_future['yhat'].sum()
                }
            else:
                results[item] = {'error': 'Forecast date must be after last training date'}
                
        except Exception as e:
            results[item] = {'error': str(e)}
        
        progress_bar.progress((idx + 1) / len(items))
    
    status_text.text("Batch processing complete!")
    progress_bar.empty()
    status_text.empty()
    
    return results


def export_batch_results_to_excel(batch_results, batch_by):
    """Export batch forecast results to Excel file"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Summary sheet
        summary_data = []
        for item, result in batch_results.items():
            if 'error' not in result:
                summary_data.append({
                    batch_by: item,
                    'Total Forecast': result['total_forecast'],
                    'MAE': result['mae'],
                    'MAE Threshold': result['threshold'],
                    'Alert Status': result['alert']
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Individual forecast sheets
        for item, result in batch_results.items():
            if 'error' not in result and 'forecast' in result:
                sheet_name = str(item)[:31]  # Excel sheet name limit
                forecast_df = result['forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                forecast_df.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                forecast_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    output.seek(0)
    return output


def run_batch_forecast_app(train):
    """New batch forecasting interface"""
    st.title("📦 Batch Forecasting by Business Segments")
    
    st.markdown("""
    Generate forecasts for multiple business segments simultaneously. This is useful for:
    - **Product Family Planning**: Forecast all product families to plan inventory
    - **Store-level Forecasting**: Generate forecasts for all stores for resource allocation
    - **Regional Analysis**: Forecast by state or city for regional planning
    - **Store Type Analysis**: Compare performance across different store types
    """)
    
    # Initialize session state
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = None
        st.session_state.batch_by = None
    
    st.sidebar.header("⚙️ Batch Forecast Settings")
    
    # Select dimension to batch by
    batch_by = st.sidebar.selectbox(
        "Batch Forecast By:",
        options=['family', 'store_nbr', 'state', 'store_type', 'city'],
        help="Select the dimension to generate individual forecasts for"
    )
    
    # Get available items
    if batch_by in train.columns:
        available_items = sorted(train[batch_by].unique())
        
        # Option to select specific items or all
        select_all = st.sidebar.checkbox("Forecast All Items", value=True)
        
        if not select_all:
            selected_items = st.sidebar.multiselect(
                f"Select {batch_by.capitalize()}:",
                options=available_items,
                default=available_items[:3] if len(available_items) >= 3 else available_items
            )
        else:
            selected_items = available_items
            st.sidebar.info(f"Will forecast {len(selected_items)} items")
    else:
        st.error(f"Column '{batch_by}' not found in data")
        return
    
    # Date selection
    last_date = train.index.max().date()
    forecast_end_date = st.sidebar.date_input(
        "Forecast End Date:",
        value=last_date + timedelta(days=30),
        min_value=last_date,
        max_value=date(2100, 1, 1)
    )
    
    periods = (pd.to_datetime(forecast_end_date) - pd.to_datetime(last_date)).days
    st.sidebar.success(f"Forecasting **{periods}** days ahead")
    
    # Run batch forecast
    if st.sidebar.button("🚀 Run Batch Forecast", type="primary"):
        if not selected_items:
            st.warning("Please select at least one item to forecast")
            return
        
        with st.spinner(f'Generating forecasts for {len(selected_items)} items...'):
            batch_results = generate_batch_forecasts(
                train, 
                forecast_end_date, 
                batch_by, 
                selected_items
            )
            st.session_state.batch_results = batch_results
            st.session_state.batch_by = batch_by
        
        st.success(f"✅ Batch forecast completed for {len(batch_results)} items!")
    
    # Display results
    if st.session_state.batch_results:
        batch_results = st.session_state.batch_results
        batch_by = st.session_state.batch_by
        
        st.markdown("---")
        st.header("📊 Batch Forecast Results")
        
        # Summary metrics
        successful = sum(1 for r in batch_results.values() if 'error' not in r)
        failed = len(batch_results) - successful
        alerts = sum(1 for r in batch_results.values() if 'alert' in r and 'ALERT' in r['alert'])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Items", len(batch_results))
        col2.metric("Successful", successful, delta=None)
        col3.metric("Failed", failed, delta=None if failed == 0 else f"-{failed}", delta_color="inverse")
        col4.metric("⚠️ Alerts", alerts, delta=None if alerts == 0 else f"-{alerts}", delta_color="inverse")
        
        # Summary table
        st.subheader("Summary Table")
        summary_data = []
        for item, result in batch_results.items():
            if 'error' in result:
                summary_data.append({
                    batch_by.capitalize(): item,
                    'Status': '❌ Failed',
                    'Error': result['error'],
                    'Total Forecast': 0,
                    'MAE': None,
                    'Alert': 'N/A'
                })
            else:
                summary_data.append({
                    batch_by.capitalize(): item,
                    'Status': '✅ Success',
                    'Error': None,
                    'Total Forecast': f"{result['total_forecast']:,.0f}",
                    'MAE': f"{result['mae']:,.2f}" if result['mae'] else 'N/A',
                    'Alert': '🚨' if 'ALERT' in result['alert'] else '✅'
                })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Export to Excel
        st.subheader("📥 Export Results")
        excel_data = export_batch_results_to_excel(batch_results, batch_by)
        st.download_button(
            label="Download Excel Report",
            data=excel_data,
            file_name=f"batch_forecast_{batch_by}_{pd.Timestamp('now').strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Detailed view for individual items
        st.markdown("---")
        st.subheader("🔍 Detailed Forecast View")
        
        # Filter successful forecasts
        successful_items = [item for item, r in batch_results.items() if 'error' not in r]
        
        if successful_items:
            selected_item = st.selectbox(
                f"Select {batch_by.capitalize()} to View:",
                options=successful_items
            )
            
            if selected_item:
                result = batch_results[selected_item]
                
                # Alert status
                if 'ALERT' in result['alert']:
                    st.error(f"🚨 **{result['alert']}** - MAE: {result['mae']:,.2f} | Threshold: {result['threshold']:,.2f}")
                else:
                    st.info(f"✅ **{result['alert']}** - MAE: {result['mae']:,.2f} | Threshold: {result['threshold']:,.2f}")
                
                # Forecast table
                with st.expander("📋 View Forecast Table", expanded=False):
                    forecast_display = result['forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                    forecast_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                    st.dataframe(forecast_display, use_container_width=True, hide_index=True)
                
                # Visualization
                st.subheader("📈 Forecast Chart")
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=result['historical']['ds'],
                    y=result['historical']['y'],
                    mode='markers',
                    name='Historical',
                    marker=dict(color='blue', size=4)
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=result['full_forecast']['ds'],
                    y=result['full_forecast']['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#1abc9c', width=2)
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=pd.concat([result['full_forecast']['ds'], result['full_forecast']['ds'].iloc[::-1]]),
                    y=pd.concat([result['full_forecast']['yhat_upper'], result['full_forecast']['yhat_lower'].iloc[::-1]]),
                    fill='toself',
                    fillcolor='rgba(27, 188, 156, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval'
                ))
                
                fig.update_layout(
                    title=f'Forecast for {batch_by.capitalize()}: {selected_item}',
                    xaxis_title='Date',
                    yaxis_title='Sales',
                    title_font_size=18
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No successful forecasts to display")


def run_forecast_app(model, prophet_df):
    st.title("📈 Time Series Forecasting (Prophet)")
    
    if 'forecast_data' not in st.session_state:
        st.session_state.forecast_data = None
        st.session_state.forecast_future_data = None
        st.session_state.model_fit = None
        st.session_state.mae = None
        st.session_state.mae_threshold_value = None
        st.session_state.alert_status = None
        st.session_state.mae_percent_threshold = None

    if model is None or prophet_df.empty:
        st.error("Prophet model or historical data is missing. Cannot run forecast.")
        return

    st.sidebar.header("Forecast Settings")
    
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
        st.sidebar.info(f"The selected end date is in the past or current date. Prophet will show historical fit.")
        
    if st.sidebar.button("🚀 Run Forecast", key='forecast_button'):
        if periods > 0:
            with st.spinner('Generating forecast and checking performance...'):
                future = model.make_future_dataframe(periods=periods)
                forecast = model.predict(future)
                
                last_train_date_only = last_train_date.date()
                forecast_future = forecast[forecast['ds'].dt.date > last_train_date_only].copy()
                
                st.session_state.forecast_data = forecast
                st.session_state.forecast_future_data = forecast_future
                st.session_state.model_fit = model 
                
                MAE_PERCENT_THRESHOLD = 0.25
                mae_result, threshold_value, alert_status = check_model_performance(prophet_df, forecast, MAE_PERCENT_THRESHOLD)
                
                st.session_state.mae = mae_result
                st.session_state.mae_threshold_value = threshold_value
                st.session_state.alert_status = alert_status
                st.session_state.mae_percent_threshold = MAE_PERCENT_THRESHOLD

            st.success("Forecast generated and performance checked successfully!")
        else:
            st.error("Forecast end date must be after the last training date to run a prediction.")

    if st.session_state.forecast_data is not None:
        forecast_data = st.session_state.forecast_data
        forecast_future = st.session_state.forecast_future_data
        model_fit = st.session_state.model_fit
        
        if 'mae' in st.session_state and st.session_state.mae is not None:
            st.markdown("---")
            st.subheader("Model Performance & Alert System 🔔")

            mae_val = st.session_state.mae
            alert_status = st.session_state.alert_status
            threshold_value = st.session_state.mae_threshold_value
            percent_threshold = st.session_state.mae_percent_threshold * 100
            
            if "ALERT" in alert_status:
                st.error(f"**🚨 {alert_status}**\n\n**Action Required:** Prediction accuracy has dropped below the defined threshold. Stakeholders should be notified. \n\n*Threshold: < {percent_threshold:.0f}% of Mean Sales (Calculated Value: {threshold_value:,.2f} MAE)*")
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
        
        st.header("Forecast Results")
        
        st.subheader("Forecast Table (Future Days)")
        if not forecast_future.empty:
            st.dataframe(
                forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
                    columns={'ds': 'Date', 'yhat': 'Forecast Value', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'}
                ).set_index('Date').style.format({'Forecast Value': "{:,.0f}", 'Lower Bound': "{:,.0f}", 'Upper Bound': "{:,.0f}"}),
                use_container_width=True
            )
        else:
            st.warning("No future dates found in the forecast data. Check the selected end date.")

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

        # Confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_data['ds'], forecast_data['ds'].iloc[::-1]]),
            y=pd.concat([forecast_data['yhat_upper'], forecast_data['yhat_lower'].iloc[::-1]]),
            fill='toself',
            fillcolor='rgba(27, 188, 156, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))

        fig.update_layout(
            title="Forecast vs Actual Sales",
            xaxis_title="Date",
            yaxis_title="Sales",
            title_font_size=18
        )

        st.plotly_chart(fig, use_container_width=True)


# --- Main App Navigation ---

        train, min_date, max_date, sort_state, prophet_df = load_data()
        model = load_prophet_model(MODEL_PATH)
        
        if model is None:
            model = train_prophet_model(prophet_df)
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(model, f)
        
        st.sidebar.title("App Navigation")
        mode = st.sidebar.selectbox("Choose Mode", ["Dashboard", "Forecast", "Batch Forecast"])
        
        if mode == "Dashboard":
            run_dashboard(train, min_date, max_date, sort_state)
        elif mode == "Forecast":
            run_forecast_app(model, prophet_df)
        elif mode == "Batch Forecast":
            run_batch_forecast_app(train)
