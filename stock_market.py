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

       # Prepare input for prediction
last_10_days = X_scaled[-10:]
next_30_days = []

for _ in range(30):
    input_data = np.reshape(last_10_days, (1, last_10_days.shape[0], 1))
    next_day_prediction = model.predict(input_data)
    next_30_days.append(next_day_prediction[0, 0])
    last_10_days = np.roll(last_10_days, -1, axis=0)
    last_10_days[-1] = next_day_prediction

# Plot future predictions
future_dates = pd.date_range(start=data.index[-1], periods=31, freq='D')[1:]
future_prices = scaler.inverse_transform([[0] * (X.shape[1] - 1) + [price] for price in next_30_days])[:, -1]

fig, ax = plt.subplots()
ax.plot(data.index, data['Close'], label="Historical Prices")
ax.plot(future_dates, future_prices, label="Future Predictions", linestyle='--', color="green")
ax.set_title("Stock Price Prediction for Next 30 Days")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)


     

        # Identify trend
        last_known_price = data['Close'].values[-1]
        trend = "Increasing" if future_predictions[-1] > last_known_price else "Decreasing"
        st.write(f"**Trend Prediction**: The stock price is likely to **{trend}** over the next 30 days.")

        # Show final predicted prices
        st.write("**Predicted Prices:**")
        st.dataframe(pd.DataFrame(future_predictions, columns=["Predicted Price"]))
