import MetaTrader5 as mt
import pandas as pd
import plotly.express as px
from datetime import datetime
from xgboost import XGBRegressor
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU
import numpy as np

mt.initialize()

login=5005258496
password="cdjwa0pl"
server="MetaQuotes-Demo"

mt.login(login,password,server)

account_info=mt.account_info()
#print(account_info)

data1 = pd.DataFrame(mt.copy_rates_range("EURUSD",mt.TIMEFRAME_D1,datetime(2022,1,1), datetime.now()))
data2 = pd.DataFrame(mt.copy_rates_range("GBPUSD",mt.TIMEFRAME_D1,datetime(2022,1,1), datetime.now()))
data3 = pd.DataFrame(mt.copy_rates_range("USDCHF",mt.TIMEFRAME_D1,datetime(2022,1,1), datetime.now()))

def moving_average(df,period=20):
    #df["ma"]=df["close"].rolling(period).mean()
    df["ma"]=df["close"].ewm(span=period,adjust=False).mean()
    return df

data1=moving_average(data1)
data2=moving_average(data2)
data3=moving_average(data3)

data1_ma=data1[["ma"]].copy()
#data1_rsi["target"]=data1_rsi.rsi.shift(-1)
data1_ma.dropna(inplace=True)

data2_ma=data2[["ma"]].copy()
#data2_rsi["target"]=data1_rsi.rsi.shift(-1)
data2_ma.dropna(inplace=True)

data3_ma=data3[["ma"]].copy()
#data1_rsi["target"]=data1_rsi.rsi.shift(-1)
data3_ma.dropna(inplace=True)

scaler = MinMaxScaler(feature_range=(0,1))

print("Data1")
scaled_data = scaler.fit_transform(data1_ma['ma'].values.reshape(-1,1))
#scaled_data = scaled_data1.copy()


prediction_days = 10

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x,0])
    y_train.append(scaled_data[x,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(GRU(3, return_sequences=True, input_shape=(x_train.shape[1], 1) ))
model.add(GRU(3, return_sequences=False))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs = 50, batch_size =32)

model.save("../models/moving_average/rnn/EURUSD.h5")

real_data = [x_train[len(x_train + 1) - prediction_days:len(x_train),0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
print(real_data)
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(prediction)



print("Data2")
scaled_data = scaler.fit_transform(data2_ma['ma'].values.reshape(-1,1))
#scaled_data = scaled_data1.copy()


prediction_days = 10

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x,0])
    y_train.append(scaled_data[x,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(GRU(3, return_sequences=True, input_shape=(x_train.shape[1], 1) ))
model.add(GRU(3, return_sequences=False))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs = 50, batch_size =32)

model.save("../models/moving_average/rnn/GBPUSD.h5")

real_data = [x_train[len(x_train + 1) - prediction_days:len(x_train),0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
print(real_data)
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(prediction)


print("Data3")
scaled_data = scaler.fit_transform(data3_ma['ma'].values.reshape(-1,1))
#scaled_data = scaled_data1.copy()


prediction_days = 10

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x,0])
    y_train.append(scaled_data[x,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(GRU(3, return_sequences=True, input_shape=(x_train.shape[1], 1) ))
model.add(GRU(3, return_sequences=False))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs = 50, batch_size =32)

model.save("../models/moving_average/rnn/USDCHF.h5")

real_data = [x_train[len(x_train + 1) - prediction_days:len(x_train),0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
print(real_data)
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(prediction)
