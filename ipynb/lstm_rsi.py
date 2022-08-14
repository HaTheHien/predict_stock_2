import MetaTrader5 as mt
import pandas as pd
import plotly.express as px
from datetime import datetime
from xgboost import XGBRegressor
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
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
# fig = px.line(data1, x=data1['time'], y = data1['close'])

scaler = MinMaxScaler(feature_range=(0,1))

def rsi(df,periods=14,ema=True):
    # delta=df["close"].diff()
    # delta.dropna(inplace=True)
    # change_up=delta.copy()
    # change_down=delta.copy()
    # change_up[change_up<0]=0
    # change_down[change_down>0]=0
    # avg_up=change_up.rolling(n).mean()
    # avg_down=change_down.rolling(n).mean().abs()
    # rsi=100*(avg_up/(avg_up+avg_down))
    # df["rsi"]=rsi
    # return df

    close_delta = df['close'].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema == True:
	    # Use exponential moving average
        ma_up = up.ewm(com = periods, adjust=False).mean()
        ma_down = down.ewm(com = periods, adjust=False).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = periods, adjust=False).mean()
        ma_down = down.rolling(window = periods, adjust=False).mean()
        
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    df["rsi"]=rsi
    return df

data1=rsi(data1)
data2=rsi(data2)
data3=rsi(data3)

data1_rsi=data1[["rsi"]].copy()
#data1_rsi["target"]=data1_rsi.rsi.shift(-1)
data1_rsi.dropna(inplace=True)

data2_rsi=data2[["rsi"]].copy()
#data1_rsi["target"]=data1_rsi.rsi.shift(-1)
data2_rsi.dropna(inplace=True)

data3_rsi=data3[["rsi"]].copy()
#data1_rsi["target"]=data1_rsi.rsi.shift(-1)
data3_rsi.dropna(inplace=True)

print("Data 1")
scaled_data = scaler.fit_transform(data1_rsi['rsi'].values.reshape(-1,1))
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
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs = 50, batch_size =32, verbose=0)

model.save("../models/rsi/lstm/EURUSD.h5")

real_data = [x_train[len(x_train + 1) - prediction_days:len(x_train),0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
print(real_data)
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(prediction)
print(data1)



print("Data 2")
scaled_data = scaler.fit_transform(data2_rsi['rsi'].values.reshape(-1,1))
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
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs = 50, batch_size =32, verbose=0)

model.save("../models/rsi/lstm/GBPUSD.h5")

real_data = [x_train[len(x_train + 1) - prediction_days:len(x_train),0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
print(real_data)
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(prediction)
print(data2)



print("Data 3")
scaled_data = scaler.fit_transform(data3_rsi['rsi'].values.reshape(-1,1))
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
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs = 50, batch_size =32, verbose=0)

model.save("../models/rsi/lstm/USDCHF.h5")

real_data = [x_train[len(x_train + 1) - prediction_days:len(x_train),0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
print(real_data)
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(prediction)
print(data3)