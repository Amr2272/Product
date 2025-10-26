import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

@st.cache_data
def load_data():
    """
    Attempts to load data from 'Data.zip'. If unsuccessful, generates mock data.
    """
    try:
        # Attempt to load user data file
        train = pd.read_csv('Data.zip')
        st.success("Data loaded successfully from Data.zip.")
    except FileNotFoundError:
        # Generate mock data if file not found
        st.warning("File 'Data.zip' not found. Generating mock data for display.")
        
        # --- Mock Data Generation based on provided image columns ---
        dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
        n_days = len(dates)
        
        states = ['Pichincha', 'Guayas', 'Azuay', 'Manabi', 'El Oro']
        store_types = ['A', 'B', 'C', 'D']
        families = ['AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS', 'BREAD/BAKERY', 'CLEANING', 'DAIRY'] # Example families
        
        # Generate enough data for a meaningful mock
        mock_data = []
        record_id = 0
        for day in dates:
            for store_nbr in range(1, 6): # 5 example stores
                state = np.random.choice(states)
                
                # Assign cities based on state
                if state == 'Pichincha':
                    city = np.random.choice(['Quito', 'Rumiñahui'])
                elif state == 'Guayas':
                    city = np.random.choice(['Guayaquil', 'Daule'])
                elif state == 'Azuay':
                    city = np.random.choice(['Cuenca'])
                else:
                    city = np.random.choice(['City X', 'City Y'])
                
                store_type = np.random.choice(store_types)
                cluster = np.random.randint(1, 18) # Example clusters 1-17
                
                for family in families:
                    sales = np.random.randint(0, 500) if np.random.rand() > 0.1 else 0 # Simulate some zero sales
                    onpromotion = np.random.randint(0, 50) if sales > 0 else 0
                    dcoilwtico = round(np.random.uniform(30, 100), 2)
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
    
    # --- Data Preprocessing Steps ---
    train["date"] = pd.to_datetime(train["date"], errors="coerce")
    train = train.dropna(subset=['date']) # Drop rows where date is NaT
    train = train.set_index("date")
    train.index = pd.to_datetime(train.index) # Ensure index is datetime

    min_date = train.index.min().date()
    max_date = train.index.max().date()
    
    # Calculate total sales by state for sorting dropdown, ensure it exists
    if 'state' in train.columns and 'sales' in train.columns:
        sort_state = train.groupby('state')['sales'].sum().sort_values(ascending=False)
    else:
        sort_state = pd.Series([], dtype='float64') # Empty series if columns are missing

    return train, min_date, max_date, sort_state

def run_dashboard():
    """
    Main function to run the Streamlit application dashboard.
    """
    train, min_date, max_date, sort_state = load_data()

    st.set_page_config(layout="wide", page_title="City Sales Dashboard", page_icon="📊")
    st.title("City Sales Dashboard")

    # Display the data summary with unique counts
    st.subheader("Data Summary (Unique Values)")
    
    # Metrics
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

    # Expander for detailed unique lists
    with st.expander("View Unique Lists for Categories"):
        if 'state' in train.columns:
            st.write("**Unique States:**")
            st.code(', '.join(sorted(train['state'].unique().astype(str))))
        if 'city' in train.columns:
            st.write("**Unique Cities:**")
            st.code(', '.join(sorted(train['city'].unique().astype(str))))
        if 'store_nbr' in train.columns:
            st.write("**Unique Store Numbers:**")
            st.code(', '.join(sorted(train['store_nbr'].unique().astype(str))))
        if 'family' in train.columns:
            st.write("**Unique Families:**")
            st.code(', '.join(sorted(train['family'].unique().astype(str))))
        if 'store_type' in train.columns:
            st.write("**Unique Store Types:**")
            st.code(', '.join(sorted(train['store_type'].unique().astype(str))))
        if 'cluster' in train.columns:
            st.write("**Unique Clusters:**")
            st.code(', '.join(sorted(train['cluster'].unique().astype(str))))
        if 'day_type' in train.columns:
            st.write("**Unique Day Types:**")
            st.code(', '.join(sorted(train['day_type'].unique().astype(str))))
        
    st.markdown("---")

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("Dashboard Controls")

        col_chosen = st.multiselect(
            "Choose State",
            options=sort_state.index.tolist(),
            default=['Pichincha'] if 'Pichincha' in sort_state.index else (sort_state.index.tolist()[:1] if not sort_state.empty else []),
            key='state_id',
            placeholder="Select one or more states"
        )

        agg_method = st.radio(
            "Aggregate Method",
            options=['sum', 'mean'],
            index=0,  
            horizontal=True,
            key='value_id'
        )

        st.markdown("---")
        st.subheader("Date Range Selection")

        date_range = st.date_input(
            "Select Date Range",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date,
            key='date_id'
        )

        if len(date_range) == 2:
            start_date, end_date = date_range[0], date_range[1]
        elif len(date_range) == 1:
            start_date, end_date = date_range[0], date_range[0]
        else:
            start_date, end_date = min_date, max_date

        st.caption(f"Selected range: **{start_date.strftime('%d-%m-%Y')}** → **{end_date.strftime('%d-%m-%Y')}**")

    # --- Graph Logic ---
    # Ensure 'sales' column exists before trying to filter or aggregate
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

            fig = px.bar(
                city_sales, x='city', y='sales',
                labels={'city': 'City', 'sales': f'Sales ({agg_method.capitalize()})'},
                title=f'City Sales for Selected States ({agg_method.capitalize()})',
                text=city_sales['sales'].apply(lambda x: f'{x:,.0f}'), 
                color_discrete_sequence=["#1abc9c"]
            )

            fig.update_traces(
                textposition='outside',
                hovertemplate="<b>City:</b> %{x}<br>" +
                              "<b>Sales:</b> %{y:,.2f}<br>" +
                              "<extra></extra>"
            )
            fig.update_yaxes(tickformat=".2s", title_font=dict(size=14))
            fig.update_xaxes(title_font=dict(size=14))
            fig.update_layout(
                title_font_size=20,
            )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while processing data: {str(e)}")


if __name__ == '__main__':
    run_dashboard()
