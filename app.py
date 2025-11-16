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


@st.cache_data
def load_data():
    return pd.read_csv("sales_data.csv", parse_dates=["Date"])

df = load_data()

st.title("📈 Sales Analytics & Forecasting Dashboard")
st.markdown("Explore historical sales, trends, regions, categories, and future forecasts.")

# Sidebar Filters
st.sidebar.header("Filters")

category_filter = st.sidebar.multiselect(
    "Select Category", 
    options=df["Category"].unique(),
    default=df["Category"].unique()
)

region_filter = st.sidebar.multiselect(
    "Select Region", 
    options=df["Region"].unique(),
    default=df["Region"].unique()
)

filtered_df = df[
    df["Category"].isin(category_filter) &
    df["Region"].isin(region_filter)
]

# Tabs
tab1, tab2, tab3 = st.tabs([" Sales Dashboard", " Time-Series Trends", " Forecasting"])


# TAB 1 — DASHBOARD

with tab1:
    st.subheader(" Sales Overview")

    col1, col2, col3 = st.columns(3)
    
    col1.metric("Total Sales", f"${filtered_df['Sales'].sum():,.0f}")
    col2.metric("Units Sold", f"{filtered_df['UnitsSold'].sum():,.0f}")
    col3.metric("Avg Price", f"${filtered_df['Price'].mean():.2f}")

    # Sales by Category
    fig_cat = px.bar(
        filtered_df.groupby("Category")["Sales"].sum().reset_index(),
        x="Category", y="Sales", title="Sales by Category"
    )
    st.plotly_chart(fig_cat, use_container_width=True)

    # Sales by Region
    fig_region = px.pie(
        filtered_df,
        names="Region", values="Sales",
        title="Sales Share by Region"
    )
    st.plotly_chart(fig_region, use_container_width=True)


# TAB 2 — TIME SERIES

with tab2:
    st.subheader(" Monthly Sales Trend")

    monthly = (
        filtered_df.groupby(pd.Grouper(key="Date", freq="M"))["Sales"]
        .sum()
        .reset_index()
    )

    fig_trend = px.line(monthly, x="Date", y="Sales", title="Monthly Sales Trend")
    st.plotly_chart(fig_trend, use_container_width=True)


# TAB 3 — FORECASTING
with tab3:
    st.subheader(" Sales Forecast (Prophet Model)")

    # Prepare data for Prophet
    prophet_df = monthly.rename(columns={"Date": "ds", "Sales": "y"})

    # Check if enough rows
    if prophet_df.shape[0] < 2:
        st.error("❌ Not enough data to train a forecasting model.")
        st.info("Increase your date range or remove filters.")
    else:
        # Prophet model
        model = Prophet()
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=12, freq="M")
        forecast = model.predict(future)

        st.write("### Forecast Output")
        st.dataframe(forecast.tail())

        fig_forecast = plot_plotly(model, forecast)
        st.plotly_chart(fig_forecast, use_container_width=True)
