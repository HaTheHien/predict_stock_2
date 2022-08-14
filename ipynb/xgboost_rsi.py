import MetaTrader5 as mt
import pandas as pd
import plotly.express as px
from datetime import datetime
from xgboost import XGBRegressor
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

percent1=0.2
percent2=0.2
percent3=0.2

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
data1_rsi["target"]=data1_rsi.rsi.shift(-1)
data1_rsi.dropna(inplace=True)

data2_rsi=data2[["rsi"]].copy()
data2_rsi["target"]=data2_rsi.rsi.shift(-1)
data2_rsi.dropna(inplace=True)

data3_rsi=data3[["rsi"]].copy()
data3_rsi["target"]=data3_rsi.rsi.shift(-1)
data3_rsi.dropna(inplace=True)

def train_test_split(data,percent):
    data=data.values
    n=int(len(data)*(1-percent))
    return data[:n], data[n:]


def predict_value(data,percent,path=""):
    #Train model
    train,test=train_test_split(data,percent)
    X,y=train[:,:-1], train[:,-1]
    model=XGBRegressor(objective="reg:squarederror",n_estimators=1000)
    model.fit(X,y)
    if (path):
        model.save_model(path)

    #Predict value
    predictions=[]
    for i in range(len(test)):
        val=np.array(test[i,0]).reshape(1,-1)
        pred=model.predict(val)
        predictions.append(pred[0])
    return predictions


prediction1_rsi=predict_value(data1_rsi,percent1,"../models/rsi/xgboost/EURUSD.h5")
prediction2_rsi=predict_value(data2_rsi,percent2,"../models/rsi/xgboost/GBPUSD.h5")
prediction3_rsi=predict_value(data3_rsi,percent3,"../models/rsi/xgboost/USDCHF.h5")
# print(prediction1_rsi)
# print(prediction1_sma)
#prediction1_bb=predict_value(data1_bb,percent1)
# prediction1=predict_value(data_close1,percent1)
# # prediction2=predict_value(data_close2,percent2)
# # prediction3=predict_value(data_close3,percent3)

def unit_test(data_close,percent,prediction):
    n_test=int(len(data_close)*(1-percent))
    train,valid=data_close[:n_test],data_close[n_test:]
    valid["predictions"]=prediction
    print(valid)

unit_test(data1_rsi,percent1,prediction1_rsi)
unit_test(data2_rsi,percent2,prediction2_rsi)
unit_test(data3_rsi,percent3,prediction3_rsi)

