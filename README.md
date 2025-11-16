# 📈 Sales Analytics & Forecasting Dashboard

Welcome to the **Sales Analytics & Forecasting Dashboard**, an interactive web app built with **Streamlit** to explore historical sales data, visualize trends, and forecast future weekly sales using the **Prophet** model.



## Features

### 1 Interactive Filters
- Select specific stores to analyze.
- Filter by holiday status (holiday vs non-holiday sales).
- All filters are applied in real-time.

### 2 Sales Dashboard
- Total sales across selected stores.
- Average weekly sales.
- Number of stores included in the selection.

### 3 EDA (Exploratory Data Analysis) Charts
- Weekly sales trend over time.
- Multi-metric charts including:
  - **Temperature**
  - **Fuel Price**
  - **CPI (Consumer Price Index)**
  - **Unemployment**
- Charts are interactive and color-coded by store.

### 4 Forecasting
- Weekly sales forecasting using **Prophet**.
- Visualize **yhat (predicted sales)** with confidence intervals:
  - `yhat_lower`: lower bound
  - `yhat_upper`: upper bound
- Forecast extends 12 weeks into the future.
- Automatically updates when filters are changed.



## Installation

1. Clone this repository:
<pre>bash
git clone https://github.com/yourusername/sales-forecast-dashboard.git
cd sales-forecast-dashboard</pre>

2. Create a virtual environment (optional but recommended):
  <pre>python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
</pre>

3. Install required packages:
 <pre>pip install -r requirements.txt
</pre>

4. Run the Streamlit app:
   <pre>streamlit run app.py
</pre>

## Dataset

The app uses a sales dataset with the following columns:

| Column       | Description                              |
| ------------ | ---------------------------------------- |
| Store        | Store ID                                 |
| Date         | Date of record                           |
| Weekly_Sales | Weekly sales value                       |
| Holiday_Flag | 1 if the week contains a holiday, else 0 |
| Temperature  | Temperature at store location            |
| Fuel_Price   | Fuel price during the week               |
| CPI          | Consumer Price Index                     |
| Unemployment | Unemployment rate at the store location  |

Rows: 6,436
Time Range: Multiple years


## Live Demo

Check the app live on Streamlit Cloud:
[https://your-streamlit-link-here.streamlit.app](https://sales-forecast-dashboard-lbqwqv4m8petjdkqhrhjih.streamlit.app/)


## Technologies Used

-Python 3.x

-Streamlit

-Pandas

-Plotly Express

-Prophet (Facebook Prophet)

-Matplotlib


## Author
ABDULMUIZ SHITTU
