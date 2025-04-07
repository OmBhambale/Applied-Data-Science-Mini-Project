# Retail Analytics and Forecasting System using Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("retail_sales_dataset.csv", parse_dates=['Date'])
    df['Total_Sales'] = df['Quantity'] * df['Price per Unit']
    return df

df = load_data()

st.title("Retail Store Analytics & Forecasting Dashboard")
st.markdown("---")

# Sidebar filters
st.sidebar.header("Filter Data")
category = st.sidebar.multiselect("Select Product Category", df['Product Category'].unique(), default=df['Product Category'].unique())
start_date = st.sidebar.date_input("Start Date", df['Date'].min().date())
end_date = st.sidebar.date_input("End Date", df['Date'].max().date())

df_filtered = df[(df['Product Category'].isin(category)) & (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]

# KPIs
st.subheader("Key Performance Indicators")
col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"₹{df_filtered['Total_Sales'].sum():,.0f}")
col2.metric("Total Quantity Sold", f"{df_filtered['Quantity'].sum():,.0f}")
col3.metric("Total Transactions", f"{df_filtered.shape[0]}")

st.markdown("---")

# Visualizations
st.subheader("Sales Overview")
col1, col2 = st.columns(2)

# Bar Chart: Sales by Category
with col1:
    sales_by_category = df_filtered.groupby('Product Category')['Total_Sales'].sum().sort_values(ascending=False)
    st.plotly_chart(px.bar(sales_by_category, title="Sales by Product Category"))

# Pie Chart: Sales by Gender
with col2:
    gender_sales = df_filtered.groupby('Gender')['Total_Sales'].sum()
    st.plotly_chart(px.pie(values=gender_sales.values, names=gender_sales.index, title="Sales by Gender"))

# Line Chart: Sales over Time
st.subheader("Sales Over Time")
daily_sales = df_filtered.groupby('Date')['Total_Sales'].sum()
st.plotly_chart(px.line(daily_sales, title="Daily Sales Trend"))

# Customer Segmentation (KMeans Clustering)
st.subheader("Customer Segmentation")
cust_df = df_filtered.groupby('Customer ID').agg({
    'Total_Sales': 'sum',
    'Quantity': 'sum',
    'Age': 'mean'
}).dropna()

scaler = StandardScaler()
cust_scaled = scaler.fit_transform(cust_df)
kmeans = KMeans(n_clusters=4, random_state=42)
cust_df['Cluster'] = kmeans.fit_predict(cust_scaled)
st.plotly_chart(px.scatter(cust_df, x='Total_Sales', y='Quantity', color='Cluster', title="Customer Segments"))

# Time Series Forecasting (ARIMA)
st.subheader("Sales Forecasting using ARIMA")
df_ts = daily_sales.asfreq('D').fillna(0)
model = ARIMA(df_ts, order=(5,1,0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=15)

fig, ax = plt.subplots(figsize=(10, 4))
df_ts[-60:].plot(ax=ax, label='Historical')
forecast.plot(ax=ax, label='Forecast', color='red')
plt.title("Next 15 Days Forecast")
plt.legend()
st.pyplot(fig)

# Evaluation
mae = mean_absolute_error(df_ts[-15:], forecast[:15])
rmse = np.sqrt(mean_squared_error(df_ts[-15:], forecast[:15]))
st.markdown(f"**MAE**: {mae:.2f}, **RMSE**: {rmse:.2f}")