import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Sales Forecasting Dashboard")
st.markdown("Explore historical sales, trends, and future forecasts with Prophet.")

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("sales_data.csv", parse_dates=["Date"])
    return df

df = load_data()

# -------------------------
# Sidebar Filters
# -------------------------
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

date_min = df["Date"].min()
date_max = df["Date"].max()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max
)

filtered_df = df[
    (df["Store"].isin(store_filter)) &
    (df["Holiday_Flag"].isin(holiday_filter)) &
    (df["Date"].between(date_range[0], date_range[1]))
]

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Time-Series Trends", "Forecasting"])

# -------------------------
# TAB 1: Overview
# -------------------------
with tab1:
    st.subheader("Sales Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"${filtered_df['Weekly_Sales'].sum():,.0f}")
    col2.metric("Average Weekly Sales", f"${filtered_df['Weekly_Sales'].mean():,.0f}")
    col3.metric("Maximum Weekly Sales", f"${filtered_df['Weekly_Sales'].max():,.0f}")

    st.subheader("Sales by Store")
    fig_store = px.bar(
        filtered_df.groupby("Store")["Weekly_Sales"].sum().reset_index(),
        x="Store", y="Weekly_Sales", title="Total Sales by Store"
    )
    st.plotly_chart(fig_store, use_container_width=True)

    st.subheader("Sales by Holiday Flag")
    fig_holiday = px.pie(
        filtered_df,
        names="Holiday_Flag", values="Weekly_Sales",
        title="Sales Share by Holiday Flag"
    )
    st.plotly_chart(fig_holiday, use_container_width=True)

# -------------------------
# TAB 2: Time-Series
# -------------------------
with tab2:
    st.subheader("Weekly Sales Trend")
    weekly = (
        filtered_df.groupby(pd.Grouper(key="Date", freq="W"))["Weekly_Sales"]
        .sum()
        .reset_index()
    )
    fig_trend = px.line(weekly, x="Date", y="Weekly_Sales", title="Weekly Sales Trend")
    st.plotly_chart(fig_trend, use_container_width=True)

# -------------------------
# TAB 3: Forecasting
# -------------------------
with tab3:
    st.subheader("Sales Forecast (Prophet Model)")

    # Prepare data for Prophet
    prophet_df = weekly.rename(columns={"Date": "ds", "Weekly_Sales": "y"})

    if prophet_df.shape[0] < 2:
        st.error("❌ Not enough data to train a forecasting model.")
        st.info("Adjust filters or increase date range.")
    else:
        # Fit Prophet model
        model = Prophet()
        model.fit(prophet_df)

        # Forecast next 12 weeks
        future = model.make_future_dataframe(periods=12, freq="W")
        forecast = model.predict(future)

        # Display forecast table
        st.write("### Forecast Output (Next 12 Weeks)")
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(12))

        # Download forecast CSV
        st.download_button(
            label="Download Forecast CSV",
            data=forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(index=False),
            file_name="sales_forecast.csv",
            mime="text/csv"
        )

        # Forecast plot with shaded uncertainty
        fig_forecast = plot_plotly(model, forecast)
        st.plotly_chart(fig_forecast, use_container_width=True)
