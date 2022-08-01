#!/usr/bin/env python
# coding: utf-8
# stock_web.py

from http import server
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import numpy as np
from src.settings import LOGIN, PASSWORD, SERVER
import plotly.graph_objects as go

print(LOGIN, PASSWORD, SERVER)
try:
    import MetaTrader5 as mt

    mt.initialize()
    login = LOGIN
    password = PASSWORD
    server_name = SERVER

    mt.login(login, password, server_name)

    account_info = mt.account_info()
    print(account_info)
    df_stock = pd.DataFrame(mt.copy_rates_range("EURUSD", mt.TIMEFRAME_D1, datetime(2022, 1, 1), datetime.now()))
except ImportError:
    df_stock = pd.read_csv("../data/EURUSD_D1.csv")



app = dash.Dash()
server = app.server
scaler = MinMaxScaler(feature_range=(0,1))

def switchTime(time):
    if time == "1 Week":
        return mt.TIMEFRAME_W1
    if time == "1 Hour":
        return mt.TIMEFRAME_H1
    return mt.TIMEFRAME_D1

def getPathModel(predictType, modelType, price):  #modelType: LSTM, RNN; predictType: Closing
    return "./models/" + predictType + "/" + modelType + "/" + price + ".h5"

def getData(time, price):
    _time = switchTime(time)
    data = pd.DataFrame(mt.copy_rates_range(price, _time, datetime(2022, 1, 1), datetime.now()))
    return data

def getGraphCandle(data):
    fig = go.Figure(go.Candlestick(
        x=data['time'],
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close']
    ))
    return fig

def getActualGraph(data):
    return {
        "data": [
            go.Scatter(
                x=data['time'],
                y=data["close"],
                mode="lines",
            )
        ],
        "layout": go.Layout(
            title='Line graph',
            xaxis={'title': 'Timestamp'},
            yaxis={'title': 'Closing Rate'}
        )
    }

app.layout = html.Div([
    dcc.Interval(id='my-interval', interval=240000),
    
    html.H1(
        "Currency Exchange Rate Prediction Analysis Dashboard",
        style={"textAlign": "center"}
    ),

    html.Div([
                html.H2(
                    "Line graph",
                    style={"textAlign": "center"}
                ),
                html.Div([
                    dcc.Dropdown(['EURUSD', 'GBPUSD', 'USDCHF'], 'EURUSD', id='price-dropdown'),
                    dcc.Dropdown(['1 Day', '1 Week', '1 Hour'], '1 Day', id='time-dropdown'),
                    dcc.Dropdown(['LSTM', 'RNN', 'XGBOOST'], 'LSTM', id='predict-type-dropdown'),
                ]),
                dcc.Loading(
                    id="ls-loading-2",
                    children=[
                        html.H2(
                            id="predictClosing",
                            style={"textAlign": "center"}
                        ),
                        html.H2(
                            id="predictPriceOfChange",
                            style={"textAlign": "center"}
                        )
                    ],
                    type="circle",
                ),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data": [
                            go.Scatter(
                                x=df_stock['time'],
                                y=df_stock["close"],
                                mode="lines",
                            )
                        ],
                        "layout": go.Layout(
                            title='Line graph',
                            xaxis={'title': 'Timestamp'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                ),
            ]),

    html.Div([
                html.H2(
                    "Candle graph",
                    style={"textAlign": "center"}
                ),
                dcc.Graph(id="Candle graph", figure=go.Figure(go.Candlestick(
                    x=df_stock['time'],
                    open=df_stock['open'],
                    high=df_stock['high'],
                    low=df_stock['low'],
                    close=df_stock['close']
                ))),
            ]),
])

#predict data
@app.callback(
    Output("predictClosing","children")
    ,Output("predictPriceOfChange","children"),
    Input('my-interval', 'n_intervals') #get data with 1 interval
    ,Input('time-dropdown', 'value')
    ,Input('predict-type-dropdown', 'value')
    ,Input('price-dropdown', 'value'))
def multi_output(n_intervals, time, predictType, price):
    prediction_days = 60
    #get data
    data = getData(time,price)

    #closing model
    scaled_data_closing = scaler.fit_transform(data['close'].values.reshape(-1,1))
    modelClosing = load_model(getPathModel("Closing",predictType, price))

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data_closing)):
        x_train.append(scaled_data_closing[x-prediction_days:x,0])
        y_train.append(scaled_data_closing[x,0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    real_data = [x_train[len(x_train + 1) - prediction_days:len(x_train),0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = modelClosing.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(prediction)

    #Price of change
    scaled_data_price_of_change = scaler.fit_transform((data['high']-data['low']).values.reshape(-1,1))
    modelPriceOfChange = load_model(getPathModel("Price_of_change",predictType, price))

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data_price_of_change)):
        x_train.append(scaled_data_price_of_change[x-prediction_days:x,0])
        y_train.append(scaled_data_price_of_change[x,0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    real_data = [x_train[len(x_train + 1) - prediction_days:len(x_train),0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction2 = modelPriceOfChange.predict(real_data)
    prediction2 = scaler.inverse_transform(prediction2)
    print(prediction2)


    return ["Predict closing: " +  str(prediction[0][0]), "Predict price of change: " + str(prediction2[0][0])]

#update graph
@app.callback(
    Output("Candle graph", "figure")
    ,Output("Actual Data", "figure"),
    Input('my-interval', 'n_intervals') #get data with 1 interval
    ,Input('time-dropdown', 'value')
    ,Input('price-dropdown', 'value'))
def multi_output(n_intervals, time, price):
    #get data
    data = getData(time,price)

    return [getGraphCandle(data), getActualGraph(data)]

if __name__ == '__main__':
    app.run(debug=True)
