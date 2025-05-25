import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime

# Set page configuration
st.set_page_config(layout="wide", page_title="Gold Price Forecast")

# --- Load Data ---
@st.cache_data
def load_data():
    """
    Loads the gold price data from Gold_data.csv.
    Caches the data to improve performance.
    """
    try:
        df = pd.read_csv("Gold_data.csv")
        # Convert 'date' column to datetime objects
        df['date'] = pd.to_datetime(df['date'])
        # Set 'date' as the DataFrame index
        df.set_index('date', inplace=True)
        # Ensure the 'price' column is numeric
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        # Drop any rows with NaN values that might result from coercion
        df.dropna(subset=['price'], inplace=True)
        return df
    except FileNotFoundError:
        st.error("Error: Gold_data.csv not found. Please ensure the file is in the same directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return pd.DataFrame()

# --- Feature Engineering ---
def create_features(df, lag=1):
    """
    Creates lagged features for the 'price' column.
    """
    df_copy = df.copy()
    # Create a lagged feature (e.g., price of the previous day)
    df_copy['price_lag1'] = df_copy['price'].shift(lag)
    # Drop rows with NaN values resulting from the shift
    df_copy.dropna(inplace=True)
    return df_copy

# --- Model Training and Prediction ---
def train_and_forecast(df, forecast_days=30):
    """
    Trains a Linear Regression model and forecasts future gold prices.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), None

    # Create features
    df_features = create_features(df)

    # Define features (X) and target (y)
    X = df_features[['price_lag1']]
    y = df_features['price']

    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Prepare for forecasting
    last_known_price = df['price'].iloc[-1]
    last_known_date = df.index[-1]

    forecast_prices = []
    current_price = last_known_price
    current_date = last_known_date

    # Generate future dates and predict prices
    for _ in range(forecast_days):
        # Predict the next day's price
        next_price_prediction = model.predict(np.array([[current_price]]))
        forecast_prices.append(next_price_prediction[0])

        # Update current price and date for the next iteration
        current_price = next_price_prediction[0]
        current_date += datetime.timedelta(days=1)

    # Create a DataFrame for the forecast
    forecast_dates = [last_known_date + datetime.timedelta(days=i+1) for i in range(forecast_days)]
    forecast_df = pd.DataFrame({'price': forecast_prices}, index=forecast_dates)
    forecast_df.index.name = 'date'

    return model, forecast_df

# --- Streamlit App Layout ---
st.title("💰 Gold Price Forecast App")
st.markdown("This application forecasts gold prices for the next 30 days using a simple Linear Regression model based on historical data.")

# Load data
gold_df = load_data()

if not gold_df.empty:
    st.header("Historical Gold Prices")
    st.write(gold_df.tail()) # Display last few historical records

    # Plot historical data
    st.subheader("Historical Gold Price Trend")
    fig_hist, ax_hist = plt.subplots(figsize=(12, 6))
    ax_hist.plot(gold_df.index, gold_df['price'], label='Historical Price', color='skyblue')
    ax_hist.set_title('Historical Gold Price Over Time')
    ax_hist.set_xlabel('Date')
    ax_hist.set_ylabel('Price')
    ax_hist.legend()
    ax_hist.grid(True)
    st.pyplot(fig_hist)

    # Train model and get forecast
    st.header("30-Day Gold Price Forecast")
    model, forecast_df = train_and_forecast(gold_df, forecast_days=30)

    if not forecast_df.empty:
        st.subheader("Forecasted Gold Prices (Next 30 Days)")
        st.write(forecast_df)

        # Combine historical and forecasted data for plotting
        combined_df = pd.concat([gold_df, forecast_df])

        st.subheader("Historical and Forecasted Gold Prices")
        fig_combined, ax_combined = plt.subplots(figsize=(12, 6))
        ax_combined.plot(gold_df.index, gold_df['price'], label='Historical Price', color='skyblue')
        ax_combined.plot(forecast_df.index, forecast_df['price'], label='Forecasted Price', color='salmon', linestyle='--')
        ax_combined.set_title('Gold Price: Historical vs. 30-Day Forecast')
        ax_combined.set_xlabel('Date')
        ax_combined.set_ylabel('Price')
        ax_combined.legend()
        ax_combined.grid(True)
        st.pyplot(fig_combined)

        st.markdown(
            """
            **Note:** This forecast is based on a simple Linear Regression model using the previous day's price.
            More sophisticated time series models (e.g., ARIMA, Prophet, LSTM) could provide more accurate predictions.
            This app is for demonstration purposes only and should not be used for financial decisions.
            """
        )
    else:
        st.warning("Could not generate a forecast. Please check the data and model.")
else:
    st.error("Unable to process gold price data. Please ensure 'Gold_data.csv' is correctly formatted and available.")

