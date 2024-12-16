import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load model
model = load_model(r'C:\Users\kesha\Project (stock analysis)\Stock Predictions Model.keras')

# Add custom CSS styling
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 40px;
        color: #2F4F4F;
        font-family: 'Arial', sans-serif;
        margin-top: 20px;
    }

    .header {
        font-size: 30px;
        color: #FF6347;
        font-family: 'Arial', sans-serif;
    }

    .subheader {
        font-size: 20px;
        color: #4682B4;
    }

    .prediction-section {
        background-color: #F0F8FF;
        padding: 10px;
        border-radius: 10px;
        margin-top: 20px;
    }

    .container {
        max-width: 900px;
        margin: 0 auto;
    }

    .input-box {
        width: 300px;
        padding: 10px;
        font-size: 16px;
        margin-bottom: 20px;
    }

    .plot-container {
        display: flex;
        justify-content: space-around;
        margin-top: 20px;
    }

    </style>
""", unsafe_allow_html=True)

# Header for the app
st.header('Stock Market Predictor')

# Input for stock symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2014-01-01'
end = '2024-11-27'

# Fetch stock data using yfinance
data = yf.download(stock, start, end)

# Display stock data
st.header('Stock Data')
st.write(data)

# Split data into train and test
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

# Scaling the data
scaler = MinMaxScaler(feature_range=(0,1))
data_train_scaled = scaler.fit_transform(data_train)

# Prepare the test data by adding the last 100 days from training data
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.transform(data_test)

# MA-50 Plot
st.subheader('MA-50')
ma_50_days = data.Close.rolling(50).mean()
fig7 = plt.figure(figsize=(12,8))
plt.plot(ma_50_days, 'r', label="MA Price")
plt.plot(data.Close, 'g', label="Original Price")
plt.legend()  # Ensure the legend is shown
st.pyplot(fig7)  # Display the plot in the Streamlit app

# MA-100 Plot
st.subheader('MA-100')
ma_100_days = data.Close.rolling(100).mean()
fig1 = plt.figure(figsize=(12,8))
plt.plot(ma_100_days, 'r', label="MA Price")
plt.plot(data.Close, 'g', label="Original Price")
plt.legend()  # Ensure the legend is shown
st.pyplot(fig1)  # Display the plot in the Streamlit app

# MA-100 vs MA-200 Plot
st.subheader('Price vs MA-100 vs MA-200')
ma_200_days = data.Close.rolling(200).mean()
fig2 = plt.figure(figsize=(12,8))
plt.plot(ma_100_days, 'r', label='MA-100 Days')
plt.plot(ma_200_days, 'b', label='MA-200 Days')
plt.plot(data.Close, 'g', label='Original Price')
plt.legend()  # Ensure the legend is shown
st.pyplot(fig2)  # Display the plot in the Streamlit app

# Prepare data for prediction
x = []
y = []

for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i-100:i])
    y.append(data_test_scaled[i, 0])

x, y = np.array(x), np.array(y)

# Predict using the model
predict = model.predict(x)

# Inverse scaling
scale = 1 / scaler.scale_[0]  # Correct the inverse scaling to match the original prices
predict = predict * scale
y = y * scale

# Plot Original vs Predicted Price
st.subheader('Original Price VS Predicted Price')
fig3 = plt.figure(figsize=(12,8))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'b', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()  # Ensure the legend is shown
st.pyplot(fig3)  # Display the plot in the Streamlit app
