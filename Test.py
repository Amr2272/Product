import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

@st.cache_data
def load_data():

    try:
   
        train = pd.read_csv('Data.zip')
        st.success("تم تحميل البيانات بنجاح من ملف Data.zip.")
    except FileNotFoundError:
        st.warning("لم يتم العثور على ملف 'Data.zip'. جاري توليد بيانات وهمية للعرض.")
        
        dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
        states = ['Pichincha', 'Guayas', 'Azuay', 'Manabi', 'El Oro']
        state_data = []

        for state in states:
            for day in dates:
                if state == 'Pichincha':
                    cities = ['Quito', 'Rumiñahui']
                elif state == 'Guayas':
                    cities = ['Guayaquil', 'Daule']
                elif state == 'Azuay':
                    cities = ['Cuenca']
                else:
                    cities = ['Other City A', 'Other City B']

                city = np.random.choice(cities)
                sales = np.random.randint(50, 500)
                state_data.append([day, state, city, sales])

        train = pd.DataFrame(state_data, columns=['date', 'state', 'city', 'sales'])

  
    train["date"] = pd.to_datetime(train["date"], errors="coerce")
    
    train = train.dropna(subset=['date'])

   
    train = train.set_index("date")
   
    train.index = pd.to_datetime(train.index)

    min_date = train.index.min().date()
    max_date = train.index.max().date()
    
   
    sort_state = train.groupby('state')['sales'].sum().sort_values(ascending=False)

    return train, min_date, max_date, sort_state

def run_dashboard():
    """
    الدالة الرئيسية لتشغيل تطبيق Streamlit.
    """
    
    train, min_date, max_date, sort_state = load_data()

    
    st.set_page_config(layout="wide", page_title="City Sales Dashboard", page_icon="📊")

   
    st.title("City Sales Dashboard: لوحة تحكم مبيعات المدن")

    
    with st.sidebar:
        st.header("Dashboard Controls | عناصر التحكم")

        
        col_chosen = st.multiselect(
            "Choose State | اختر الولاية",
            options=sort_state.index.tolist(),
            default=['Pichincha'],
            key='state_id',
            placeholder="Select one or more states"
        )


        agg_method = st.radio(
            "Aggregate | طريقة التجميع",
            options=['sum', 'mean'],
            index=0,  
            horizontal=True,
            key='value_id'
        )

        st.markdown("---")
        st.subheader("Date Range Selection | تحديد نطاق التاريخ")

        date_range = st.date_input(
            "Select Date Range | اختر نطاقًا زمنيًا",
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



    if not col_chosen:
        st.warning("رجاءً، اختر ولاية واحدة على الأقل لعرض بيانات المبيعات.")
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
            st.warning("لم يتم العثور على بيانات لنطاق التاريخ والفلاتر المحددة.")
            fig = px.bar(title="لا توجد بيانات للفلاتر المختارة")

        else:
            if agg_method == 'mean':
                city_sales = filterr.groupby('city')['sales'].mean().reset_index()
            else:
                city_sales = filterr.groupby('city')['sales'].sum().reset_index()

            fig = px.bar(
                city_sales, x='city', y='sales',
                labels={'city': 'المدينة', 'sales': f'المبيعات ({agg_method.capitalize()})'},
                title=f'مبيعات المدن للولايات المختارة ({agg_method.capitalize()})',
                text=city_sales['sales'].apply(lambda x: f'{x:,.0f}'), 
                color_discrete_sequence=["#1abc9c"]
            )

            fig.update_traces(
                textposition='outside',
                hovertemplate="<b>المدينة:</b> %{x}<br>" +
                              "<b>المبيعات:</b> %{y:,.2f}<br>" +
                              "<extra></extra>"
            )
            fig.update_yaxes(tickformat=".2s", title_font=dict(size=14))
            fig.update_xaxes(title_font=dict(size=14))
            fig.update_layout(
                title_font_size=20,
            )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة البيانات: {str(e)}")


if __name__ == '__main__':
    run_dashboard()
