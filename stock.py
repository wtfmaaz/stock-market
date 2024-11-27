
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Function to create the dataset for LSTM
def create_dataset(data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Load and preprocess data
@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

# Predict the next 30 days
def predict_future(model, data, look_back, scaler, days=30):
    predictions = []
    input_seq = data[look_back:]

    for _ in range(days):
        input_seq_reshaped = np.reshape(input_seq, (1, look_back, 1))
        next_pred = model.predict(input_seq_reshaped)
        predictions.append(next_pred[0, 0])
        input_seq = np.append(input_seq[1:], next_pred[0, 0])

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Main app
st.title("Apple Stock Price Prediction")
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("/content/AAPL (1).csv", type=["csv"])

if uploaded_file is not None:
    st.sidebar.success("Dataset Uploaded Successfully!")

    # Load data
    data = load_data(uploaded_file)

    # Split data into train and test
    train_data = data[:'2016']
    test_data = data['2019':]

    # Scaling data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_data['Close'].values.reshape(-1, 1))

    # Create train dataset
    look_back = 60
    X_train, y_train = create_dataset(scaled_data, look_back)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])

    # Compile and train
    model.compile(optimizer='adam', loss='mean_squared_error')
    with st.spinner('Training the model...'):
        model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=0)

    # Prepare test data
    total_data = pd.concat((train_data['Close'], test_data['Close']), axis=0)
    inputs = total_data[len(total_data) - len(test_data) - look_back:].values
    inputs = scaler.transform(inputs.reshape(-1, 1))

    X_test, y_test = create_dataset(inputs, look_back)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Predictions
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Plot predictions
    st.subheader("Actual vs Predicted Prices")
    plt.figure(figsize=(14, 6))
    plt.plot(test_data['Close'].values, color='blue', label='Actual Prices')
    plt.plot(predicted_prices, color='red', label='Predicted Prices')
    plt.title('Apple Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

    # Predict future
    future_predictions = predict_future(model, inputs, scaler)
    st.subheader("Next 30 Days Prediction")
    st.line_chart(future_predictions)

    # Accuracy metrics
    actual_prices = test_data['Close'].values
    accuracy = 100 - np.mean(np.abs((actual_prices[:len(predicted_prices)] - predicted_prices[:, 0]) / actual_prices[:len(predicted_prices)])) * 100
    rmse = np.sqrt(mean_squared_error(test_data['Close'].values[:len(predicted_prices)], predicted_prices))

    st.write(f"**Prediction Accuracy:** {accuracy:.2f}%")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
else:
    st.warning("Please upload a dataset to proceed.")

