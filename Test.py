import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from prophet import Prophet
from datetime import date, timedelta
import os
import plotly.graph_objects as go

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


def run_forecast_app(model, prophet_df):
    st.title("📈 Time Series Forecasting (Prophet)")
    
    if 'forecast_data' not in st.session_state:
        st.session_state.forecast_data = None
        st.session_state.forecast_future_data = None
        st.session_state.model_fit = None

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
            
            with st.spinner('Generating forecast...'):
                future = model.make_future_dataframe(periods=periods)
                forecast = model.predict(future)
                
                last_train_date_only = last_train_date.date()
                forecast_future = forecast[forecast['ds'].dt.date > last_train_date_only].copy()
                
                st.session_state.forecast_data = forecast
                st.session_state.forecast_future_data = forecast_future
                st.session_state.model_fit = model 

            st.success("Forecast generated successfully!")
        else:
            st.error("Forecast end date must be after the last training date to run a prediction.")


    if st.session_state.forecast_data is not None:
        
        forecast_data = st.session_state.forecast_data
        forecast_future = st.session_state.forecast_future_data
        model_fit = st.session_state.model_fit
        
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
        
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model Components")
        fig_components = model_fit.plot_components(forecast_data)
        st.write(fig_components)
        
    else:
        pass


if __name__ == '__main__':
    train, min_date, max_date, sort_state, prophet_df = load_data()
    model = load_prophet_model(MODEL_PATH)
    
    st.sidebar.title("App Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose App Mode",
        ["City Sales Dashboard", "Time Series Forecast"]
    )

    if app_mode == "City Sales Dashboard":
        if 'forecast_data' in st.session_state:
            st.session_state.forecast_data = None
        run_dashboard(train, min_date, max_date, sort_state)
    elif app_mode == "Time Series Forecast":
        run_forecast_app(model, prophet_df)
