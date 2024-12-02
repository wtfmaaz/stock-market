import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import streamlit as st

# Configure Streamlit page
st.set_page_config(page_title="Apple Stock Trend Prediction", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

def preprocess_data(data, feature='Close', look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[feature].values.reshape(-1, 1))

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def visualize_trend(predicted_prices, title="Next 30 Days Stock Price Prediction"):
    plt.figure(figsize=(14, 5))
    plt.plot(predicted_prices, color='green', label='Predicted Prices')
    plt.title(title)
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    st.pyplot(plt)

# Streamlit interface
st.title("Apple Stock Trend Prediction")
st.markdown("""
Upload a CSV file with Apple stock data to predict the stock price trend for the next 30 days.
""")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    # Load data
    data = load_data(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Preprocessing
    st.subheader("Preprocessing Data")
    look_back = st.slider("Select Look Back Period (days)", min_value=30, max_value=100, value=60)
    train_data = data[:'2018']

    X_train, y_train, scaler = preprocess_data(train_data, look_back=look_back)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    st.write("Training data prepared!")

    # Build and train the model
    st.subheader("Training the LSTM Model")
    model = build_lstm_model(input_shape=(look_back, 1))
    if st.button("Start Training"):
        with st.spinner("Training in progress..."):
            model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)
        st.success("Model trained successfully!")

        # Prepare future prediction data
        inputs = data['Close'].values[-look_back:]
        inputs_scaled = scaler.transform(inputs.reshape(-1, 1))

        X_future = []
        for i in range(30):  # Predicting the next 30 days
            X_future.append(inputs_scaled[-look_back:])
            predicted = model.predict(np.array(X_future[-1:]).reshape(1, look_back, 1))
            inputs_scaled = np.append(inputs_scaled, predicted, axis=0)

        future_predictions = scaler.inverse_transform(inputs_scaled[-30:])

        # Visualize future trend
        st.subheader("Prediction Results for Next 30 Days")
        visualize_trend(future_predictions)

        # Identify trend
        last_known_price = data['Close'].values[-1]
        trend = "Increasing" if future_predictions[-1] > last_known_price else "Decreasing"
        st.write(f"**Trend Prediction**: The stock price is likely to **{trend}** over the next 30 days.")

        # Show final predicted prices
        st.write("**Predicted Prices:**")
        st.dataframe(pd.DataFrame(future_predictions, columns=["Predicted Price"]))
