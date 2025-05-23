# app.py

import streamlit as st
import pandas as pd
import pickle

# Load model
with open("gold_price_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Gold Price Predictor", layout="centered")
st.title("🏆 Gold Price Prediction App")

st.write("Enter the previous day's gold price to predict today's price:")

# Input
price_lag1 = st.number_input("Previous Day's Gold Price", value=1800.0)

# Prediction
if st.button("Predict Today's Price"):
    input_df = pd.DataFrame([[price_lag1]], columns=["price_lag1"])
    prediction = model.predict(input_df)[0]
    st.success(f"📈 Predicted Gold Price: ${prediction:.2f}")
