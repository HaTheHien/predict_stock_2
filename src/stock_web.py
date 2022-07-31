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
from datetime import datetime
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
price = "EURUSE"
type = "Close"
time = "1 Day"

def switchTime(time):
    if time == "1 Minute":
        return mt.TIMEFRAME_M1
    if time == "1 Hour":
        return mt.TIMEFRAME_H1
    return mt.TIMEFRAME_D1

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
    dcc.Interval(id='my-interval', interval=1000000),
    
    html.H1(
        "Currency Exchange Rate Prediction Analysis Dashboard",
        style={"textAlign": "center"}
    ),

    html.Div([
                html.H2(
                    "Line graph",
                    style={"textAlign": "center"}
                ),
                dcc.Dropdown(['EURUSD', 'GBPUSD', 'USDCHF'], 'EURUSD', id='price-dropdown'),
                dcc.Dropdown(['1 Day', '1 Week', '1 Hour'], '1 Day', id='time-dropdown'),
                dcc.Dropdown(['LSTM', 'RNN', 'XGBoost'], 'LSTM', id='predict-type-dropdown'),
                dcc.Dropdown(['Close', 'Price of change'], 'Close', id='type-dropdown'),
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
                dcc.Graph(id="Candle graph"),
            ]),
])

@app.callback(
    Output("Candle graph", "figure")
    ,Output("Actual Data", "figure"),
    Input('my-interval', 'n_intervals') #get data with 1 interval
    ,Input('type-dropdown', 'value')
    ,Input('time-dropdown', 'value')
    ,Input('predict-type-dropdown', 'value')
    ,Input('price-dropdown', 'value'))
def multi_output(n_intervals, type, time, predictType, price):
    data = getData(time,price)
    print(data)

    return [getGraphCandle(data), getActualGraph(data)]

if __name__ == '__main__':
    app.run(debug=True)
