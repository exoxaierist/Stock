import math

import numpy as np
import tensorflow as tf
import yfinance as yf
import pandas_datareader as pdr
import datetime as dt
import pandas as pd

input_days = 10
output_days = 1
ticker = '035420.KS'
start_date = dt.datetime(2021,3,1)
end_date = dt.datetime(2022,8,1)

test_date_start = dt.datetime(2022,9,1)
test_date = dt.datetime(2023,1,25)

def fetch_data(ticker, interval, start, end):
    __raw_data = yf.download(ticker, interval=interval, start=start, end=end)
    # normalize data
    __raw_data = (__raw_data-__raw_data.mean()) / __raw_data.std()
    # splits data into days
    __split_data = split_dataframe(__raw_data,6)
    for _ in range(len(__split_data)%(input_days+output_days)):
        __split_data.pop(0)
    #__split_data = np.array_split(__raw_data, math.ceil(len(__raw_data)/6))
    #__filtered_split_data = [array for array in __split_data if array.shape[0] == 6]
    return __split_data

def split_dataframe(df, chunk_size = 10000):
    chunks = list()
    num_chunks = math.ceil(len(df)/chunk_size)
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

def process_data(data):
    # split data per range
    __split_data = split_dataframe(data, input_days+output_days)

    __train_input, __train_output = [], []
    for _data in __split_data:
        __train_input.append(_data[:input_days])
        new_i = []
        for i in _data[input_days:]:
            new_i.append(i.values[0])
        __train_output.append(new_i)

    return __train_input, __train_output

def build_model():
    __model = tf.keras.Sequential()
    __model.add(tf.keras.layers.Dense(128))
    __model.add(tf.keras.layers.Dense(128, activation='relu'))
    __model.add(tf.keras.layers.Dense(256, activation='relu'))
    __model.add(tf.keras.layers.Dense(64, activation='sigmoid'))
    __model.add(tf.keras.layers.Dense(6*output_days, activation='sigmoid'))
    __model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return __model

def train_model(__model, __input, __output):
    __input = np.array(__input)
    __output = np.array(__output)
    __input = tf.convert_to_tensor(__input.reshape((len(__input),input_days,6*6)))
    __output = tf.convert_to_tensor(__output)
    __model.fit(__input, __output, epochs= 100, batch_size=128)

def test_model(_model):
    test_data = fetch_data(ticker, '1h', test_date_start, test_date)
    test_input, test_output = process_data(test_data)
    predictions = []
    deviations = [0]*100
    for i in range(len(test_input)):
        predictions.append(_model.predict(tf.convert_to_tensor(np.array(test_input).reshape((len(test_input),input_days,6*6)))))
    for i in range(len(predictions)):
        for j in range(len(predictions[0][i])):
            deviations[i] += math.fabs(predictions[0][i][j]-test_output[0][i][j])
        deviations[i] = deviations[i] / len(predictions[0][i])
    print(deviations[:10])


data = fetch_data(ticker, '1h', start_date, end_date)
train_input, train_output = process_data(data)
model = build_model()
train_model(model, __input=train_input, __output=train_output)
# test_model(model)