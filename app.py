import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("sales_data.csv", parse_dates=["Date"])
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filters")

store_filter = st.sidebar.multiselect(
    "Select Store", 
    options=df["Store"].unique(),
    default=df["Store"].unique()
)

holiday_filter = st.sidebar.multiselect(
    "Holiday Flag", 
    options=df["Holiday_Flag"].unique(),
    default=df["Holiday_Flag"].unique()
)

# Convert min/max dates to Python date objects
date_min = df["Date"].min().date()
date_max = df["Date"].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max
)

# --- Filter Data ---
filtered_df = df[
    (df["Store"].isin(store_filter)) &
    (df["Holiday_Flag"].isin(holiday_filter)) &
    (df["Date"].dt.date.between(date_range[0], date_range[1]))
]

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Dashboard", "Time-Series Trends", "Forecasting"])

# --- TAB 1: DASHBOARD ---
with tab1:
    st.subheader("Sales Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"${filtered_df['Weekly_Sales'].sum():,.0f}")
    col2.metric("Average Weekly Sales", f"${filtered_df['Weekly_Sales'].mean():,.0f}")
    col3.metric("Number of Stores", f"{filtered_df['Store'].nunique()}")

    # Sales by Store
    fig_store = px.bar(
        filtered_df.groupby("Store")["Weekly_Sales"].sum().reset_index(),
        x="Store", y="Weekly_Sales",
        title="Total Sales by Store"
    )
    st.plotly_chart(fig_store, use_container_width=True)

    # Holiday vs Non-Holiday Sales
    fig_holiday = px.pie(
        filtered_df.groupby("Holiday_Flag")["Weekly_Sales"].sum().reset_index(),
        names="Holiday_Flag", values="Weekly_Sales",
        title="Sales: Holiday vs Non-Holiday"
    )
    st.plotly_chart(fig_holiday, use_container_width=True)

# --- TAB 2: TIME-SERIES ---
with tab2:
    st.subheader("Weekly Sales Trend")
    weekly = (
        filtered_df.groupby(pd.Grouper(key="Date", freq="W"))["Weekly_Sales"]
        .sum()
        .reset_index()
    )
    fig_trend = px.line(weekly, x="Date", y="Weekly_Sales", title="Weekly Sales Trend")
    st.plotly_chart(fig_trend, use_container_width=True)

# --- TAB 3: FORECASTING ---
with tab3:
    st.subheader("Sales Forecast (Prophet Model)")

    prophet_df = weekly.rename(columns={"Date": "ds", "Weekly_Sales": "y"})

    if prophet_df.shape[0] < 2:
        st.error("❌ Not enough data to train a forecasting model.")
        st.info("Adjust date range or remove filters.")
    else:
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=12, freq="W")
        forecast = model.predict(future)

        st.write("### Forecast Output")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        fig_forecast = plot_plotly(model, forecast)
        st.plotly_chart(fig_forecast, use_container_width=True)
