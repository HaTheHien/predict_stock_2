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
# fig.show()

percent1=0.2
percent2=0.2
percent3=0.2


data_close1=data1[["close"]].copy()
data_close1["target"]=data_close1.close.shift(-1)
data_close1.dropna(inplace=True)

data_close2=data2[["close"]].copy()
data_close2["target"]=data_close2.close.shift(-1)
data_close2.dropna(inplace=True)

data_close3=data3[["close"]].copy()
data_close3["target"]=data_close3.close.shift(-1)
data_close3.dropna(inplace=True)


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

prediction1=predict_value(data_close1,percent1,"../models/closing/xgboost/EURUSD.h5")
prediction2=predict_value(data_close2,percent2,"../models/closing/xgboost/GBPUSD.h5")
prediction3=predict_value(data_close3,percent3,"../models/closing/xgboost/USDCHF.h5")

def unit_test(data_close,percent,prediction):
    n_test=int(len(data_close)*(1-percent))
    train,valid=data_close[:n_test],data_close[n_test:]
    valid["predictions"]=prediction
    print(valid)

unit_test(data_close1,percent1,prediction1)
unit_test(data_close2,percent2,prediction2)
unit_test(data_close3,percent3,prediction3)

