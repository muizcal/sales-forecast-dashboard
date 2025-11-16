import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)


@st.cache_data
def load_data():
    df = pd.read_csv("sales_data.csv")
    # Clean Date column
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    return df

df = load_data()

st.title("📈 Sales Analytics & Forecasting Dashboard")
st.markdown("""
Explore historical sales, trends, and forecast future Weekly Sales.  
Use the sidebar to filter by store and holiday status.
""")

st.sidebar.header("Filters")
selected_store = st.sidebar.multiselect(
    "Select Store",
    options=df["Store"].unique(),
    default=df["Store"].unique()
)
holiday_filter = st.sidebar.multiselect(
    "Holiday Flag",
    options=df["Holiday_Flag"].unique(),
    default=df["Holiday_Flag"].unique()
)

filtered_df = df[
    (df["Store"].isin(selected_store)) &
    (df["Holiday_Flag"].isin(holiday_filter))
]


tab1, tab2, tab3 = st.tabs(["Sales Dashboard", "EDA Charts", "Forecasting"])


with tab1:
    st.subheader("Sales Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"${filtered_df['Weekly_Sales'].sum():,.0f}")
    col2.metric("Average Weekly Sales", f"${filtered_df['Weekly_Sales'].mean():,.2f}")
    col3.metric("Number of Stores", filtered_df['Store'].nunique())


with tab2:
    st.subheader("Weekly Sales Trend")
    weekly = filtered_df.groupby('Date')['Weekly_Sales'].sum().reset_index()
    fig_sales = px.line(weekly, x="Date", y="Weekly_Sales", title="Weekly Sales Over Time")
    st.plotly_chart(fig_sales, use_container_width=True)

    st.subheader("Temperature, Fuel Price, CPI, and Unemployment")
    col1, col2 = st.columns(2)
    with col1:
        fig_temp = px.line(filtered_df, x='Date', y='Temperature', color='Store', title='Temperature Over Time')
        st.plotly_chart(fig_temp, use_container_width=True)

        fig_fuel = px.line(filtered_df, x='Date', y='Fuel_Price', color='Store', title='Fuel Price Over Time')
        st.plotly_chart(fig_fuel, use_container_width=True)
    with col2:
        fig_cpi = px.line(filtered_df, x='Date', y='CPI', color='Store', title='CPI Over Time')
        st.plotly_chart(fig_cpi, use_container_width=True)

        fig_unemp = px.line(filtered_df, x='Date', y='Unemployment', color='Store', title='Unemployment Over Time')
        st.plotly_chart(fig_unemp, use_container_width=True)


with tab3:
    st.subheader("Weekly Sales Forecast (Prophet Model)")
    # Prepare Prophet data
    prophet_df = weekly.rename(columns={"Date": "ds", "Weekly_Sales": "y"})
    
    if prophet_df.shape[0] < 2:
        st.error("❌ Not enough data to train the forecast model.")
    else:
        model = Prophet()
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=12, freq='W')
        forecast = model.predict(future)

        st.write("### Forecast Output")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        fig_forecast = plot_plotly(model, forecast)
        st.plotly_chart(fig_forecast, use_container_width=True)
