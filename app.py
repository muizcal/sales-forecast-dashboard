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
    df = pd.read_csv("sales_data.csv", parse_dates=["Date"])
    df.columns = df.columns.str.strip()  # remove extra spaces
    return df

df = load_data()
st.write("Columns detected:", df.columns.tolist())

st.title("📈 Weekly Sales Dashboard")
st.markdown("Explore historical weekly sales, store performance, and forecast future trends.")

# Sidebar Filters
st.sidebar.header("Filters")
store_filter = st.sidebar.multiselect(
    "Select Store",
    options=df["Store"].unique(),
    default=df["Store"].unique()
)
holiday_filter = st.sidebar.multiselect(
    "Select Holiday Flag",
    options=df["Holiday_Flag"].unique(),
    default=df["Holiday_Flag"].unique()
)

filtered_df = df[
    (df["Store"].isin(store_filter)) &
    (df["Holiday_Flag"].isin(holiday_filter))
]

# Tabs
tab1, tab2, tab3 = st.tabs(["Sales Overview", "Time-Series Trends", "Forecasting"])

# --- TAB 1: Sales Overview ---
with tab1:
    st.subheader("Key Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Total Sales", f"${filtered_df['Weekly_Sales'].sum():,.0f}")
    col2.metric("Average Weekly Sales", f"${filtered_df['Weekly_Sales'].mean():,.2f}")

    # Sales by Store
    fig_store = px.bar(
        filtered_df.groupby("Store")["Weekly_Sales"].sum().reset_index(),
        x="Store", y="Weekly_Sales", title="Total Sales by Store"
    )
    st.plotly_chart(fig_store, use_container_width=True)

    # Holiday vs Non-Holiday Sales
    fig_holiday = px.pie(
        filtered_df.groupby("Holiday_Flag")["Weekly_Sales"].sum().reset_index(),
        names="Holiday_Flag", values="Weekly_Sales",
        title="Sales: Holiday vs Non-Holiday"
    )
    st.plotly_chart(fig_holiday, use_container_width=True)

# --- TAB 2: Time-Series Trends ---
with tab2:
    st.subheader("Weekly Sales Trend")
    weekly = filtered_df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    fig_trend = px.line(weekly, x="Date", y="Weekly_Sales", title="Weekly Sales Over Time")
    st.plotly_chart(fig_trend, use_container_width=True)

# --- TAB 3: Forecasting ---
with tab3:
    st.subheader("Sales Forecast (Prophet Model)")
    prophet_df = weekly.rename(columns={"Date": "ds", "Weekly_Sales": "y"})

    if prophet_df.shape[0] < 2:
        st.error("❌ Not enough data to train a forecasting model.")
        st.info("Ensure filters are not too restrictive.")
    else:
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=12, freq="W")
        forecast = model.predict(future)
        st.write("### Forecast Output")
        st.dataframe(forecast.tail())
        fig_forecast = plot_plotly(model, forecast)
        st.plotly_chart(fig_forecast, use_container_width=True)
