# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os

# Load data
def load_data(uploaded_file=None):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv('Gold_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'date': 'ds', 'price': 'y'})
    return df

# Forecast using Prophet
def forecast_prices(df, periods=30):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast, model

# Streamlit App
st.title("Gold Price Forecast")
st.write("This app predicts the gold prices for the next 30 days using historical data.")

# File uploader
uploaded_file = st.file_uploader("Upload your gold price CSV file (with 'date' and 'price' columns):", type=["csv"])

# Load and show data
data = load_data(uploaded_file)
st.subheader("Historical Gold Prices")
st.line_chart(data.set_index('ds')['y'])

# Forecast
forecast, model = forecast_prices(data)
st.subheader("Forecasted Gold Prices for Next 30 Days")
forecast_tail = forecast[['ds', 'yhat']].tail(30)
st.dataframe(forecast_tail.set_index('ds'))

# Plot forecast
st.subheader("Forecast Plot")
fig1 = model.plot(forecast)
st.pyplot(fig1)

# Plot forecast components
st.subheader("Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)
