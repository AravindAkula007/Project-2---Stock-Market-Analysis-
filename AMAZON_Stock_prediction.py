# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:36:06 2024

@author: align
"""

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import date, timedelta
import joblib

# Set title
st.title('AMAZON Stock Price Prediction')

# Load data
def load_data(ticker):
    data = yf.download(ticker, start="2020-01-01", end=date.today().strftime("%Y-%m-%d"))
    return data

data = load_data('AMZN')

# Load the trained LSTM model and scaler
model = load_model('amzn_lstm_model.h5')
scaler = joblib.load('scaler_amzn.save')

# Plot raw data
st.subheader('Historical Stock Price')
st.line_chart(data['Close'])

# Forecast function
def forecast_future(df, model, scaler, n_steps, n_days):
    data = df['Close'].values
    data = data.reshape(-1, 1)
    scaled_data = scaler.transform(data)
    
    future_predictions = []
    for _ in range(n_days):
        X = scaled_data[-n_steps:]
        X = X.reshape(1, X.shape[0], 1)
        
        predicted_price = model.predict(X)
        future_predictions.append(predicted_price[0, 0])
        
        scaled_data = np.append(scaled_data, predicted_price)
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    return future_predictions

# Input for number of days to predict
st.subheader('Select Prediction Date')
end_date = st.date_input('Predict until', value=date.today() + timedelta(days=7), 
                         min_value=date.today(), max_value=date.today() + timedelta(days=30))

n_days = (end_date - date.today()).days

# Forecast future prices
if st.button('Predict'):
    future_predictions = forecast_future(data, model, scaler, n_steps=60, n_days=n_days)
    
    # Plot the forecasted data
    future_dates = pd.date_range(start=date.today() + timedelta(days=1), periods=n_days)
    
    st.subheader('Forecasted Stock Price')
    fig, ax = plt.subplots()
    ax.plot(future_dates, future_predictions, color='red', label='Predicted Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
