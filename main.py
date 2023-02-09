import yfinance as yf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
import numpy as np

def get_data(ticker, start, end):
    # Fetch stock data using yfinance library
    data = yf.download(ticker, start=start, end=end)
    # Normalize the stock price data
    data['Close'] = (data['Close'] - data['Close'].mean()) / data['Close'].std()
    # Normalize the trade volume data
    data['Volume'] = (data['Volume'] - data['Volume'].mean()) / data['Volume'].std()
    return data

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=(input_shape[0], input_shape[1]), return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def split_data(data, train_ratio=0.8):
    train_size = int(data.shape[0] * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

def predict_prices(model, data, window_size=5):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, :])
        Y.append(data[i + window_size, 0])
    X = np.array(X)
    Y = np.array(Y)
    predictions = model.predict(X)
    return predictions, Y

def evaluate_model(model, test_data, window_size=5):
    # Predict future prices using the trained model
    predictions, Y = predict_prices(model, test_data, window_size)
    # Evaluate the model by computing the mean squared error
    mse = tf.keras.losses.mean_squared_error(predictions, Y).numpy()
    return mse

def simulate_trading(model, test_data):
    budget = 1000
    stock_holding = 0
    for i in range(1, test_data.shape[0]):
        # Predict the next price
        input_data = np.reshape(test_data[i-1, :], (1, test_data.shape[1], 1))
        prediction = model.predict(input_data)
        # Buy the stock if the prediction is positive and budget exists
        if prediction > 0 and budget > 0:
            stock_holding += budget / test_data[i, 0]
            budget = 0
        # Sell the stock if the prediction is negative and stock holding exists
        if prediction < 0 and stock_holding > 0:
            budget += stock_holding * test_data[i, 0]
            stock_holding = 0
    # Return the final profit
    profit = budget + stock_holding * test_data[-1, 0] - 1000
    return profit

# Define the stock ticker and the data interval
ticker = '035420.KS'
interval = '1h'
start = '2020-01-01'
end = '2023-02-01'
# Get the stock data
data = get_data(ticker, start, end)
# Prepare the data for training and testing
data = np.column_stack((data['Close'].values, data['Volume'].values))
train_data, test_data = split_data(data)
# Build and compile the model
input_shape = (train_data.shape[1], train_data.shape[1])
model = build_model(input_shape)
# Train the model
model.fit(train_data, train_data[:, 0], epochs=10, batch_size=32, verbose=0)
# Evaluate the model on the test data
mse = evaluate_model(model, test_data)
print(f'Mean Squared Error: {mse:.4f}')
# Simulate the trading using the model
profit = simulate_trading(model, test_data)
print(f'Profit: ${profit:.2f}')
