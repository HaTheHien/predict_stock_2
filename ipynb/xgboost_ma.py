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

def moving_average(df,period=20):
    #df["ma"]=df["close"].rolling(period).mean()
    df["ma"]=df["close"].ewm(span=period,adjust=False).mean()
    return df

data1=moving_average(data1)
data2=moving_average(data2)
data3=moving_average(data3)

data1_ma=data1[["ma"]].copy()
data1_ma["target"]=data1_ma.ma.shift(-1)
data1_ma.dropna(inplace=True)

data2_ma=data2[["ma"]].copy()
data2_ma["target"]=data2_ma.ma.shift(-1)
data2_ma.dropna(inplace=True)

data3_ma=data3[["ma"]].copy()
data3_ma["target"]=data3_ma.ma.shift(-1)
data3_ma.dropna(inplace=True)


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


prediction1_ma=predict_value(data1_ma,percent1,"../models/moving_average/xgboost/EURUSD.h5")
prediction2_ma=predict_value(data2_ma,percent2,"../models/moving_average/xgboost/GBPUSD.h5")
prediction3_ma=predict_value(data3_ma,percent3,"../models/moving_average/xgboost/USDCHF.h5")
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

unit_test(data1_ma,percent1,prediction1_ma)
unit_test(data2_ma,percent2,prediction2_ma)
unit_test(data3_ma,percent3,prediction3_ma)

