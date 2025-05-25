import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import datetime

# Load pre-trained model
def load_model(path='gold_price_model.pkl'):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model

# Prepare future dates for prediction
def create_future_dates(start_date, days=30):
    return pd.date_range(start=start_date, periods=days + 1, freq='D')[1:]

# Generate synthetic features for next 30 days (simple example)
def create_features(df):
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    return df[['day', 'month', 'year']]

# Streamlit App
st.title("Gold Price Forecast")
st.write("This app predicts the gold prices for the next 30 days using a trained model.")

# Upload CSV data
uploaded_file = st.file_uploader("Upload your gold price CSV (with 'date' and 'price' columns):", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    st.stop()

# Preprocess and show original data
data['date'] = pd.to_datetime(data['date'])
st.subheader("Historical Gold Prices")
st.line_chart(data.set_index('date')['price'])

# app.py
model = load_model('gold_price_model.pkl')

# Prepare next 30 days
last_date = data['date'].max()
future_dates = create_future_dates(last_date, 30)
future_df = pd.DataFrame({'date': future_dates})
features = create_features(future_df)

# Predict
predictions = model.predict(features)

# Display forecast
st.subheader("Forecasted Gold Prices for Next 30 Days")
predicted_df = pd.DataFrame({
    'date': future_dates,
    'predicted_price': predictions
})
st.dataframe(predicted_df.set_index('date'))

# Plot
st.subheader("Forecast Plot")
fig, ax = plt.subplots()
ax.plot(data['date'], data['price'], label='Historical')
ax.plot(predicted_df['date'], predicted_df['predicted_price'], label='Forecast', color='orange')
ax.set_xlabel('Date')
ax.set_ylabel('Gold Price')
ax.legend()
st.pyplot(fig)
