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

app.layout = html.Div([

    html.H1(
        "Currency Exchange Rate Prediction Analysis Dashboard",
        style={"textAlign": "center"}
    ),

    dcc.Tabs(id="tabs", children=[

        dcc.Tab(label='EUR TO USD', children=[
            html.Div([
                html.H2(
                    "Actual exchange rate",
                    style={"textAlign": "center"}
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
            ])
        ]),
        dcc.Tab(label='Candle stock Data', children=[
            html.Div([
                html.H4('Stock candlestick chart'),
                dcc.Checklist(
                    id='toggle-rangeslider',
                    options=[{'label': 'Include Rangeslider', 
                            'value': 'slider'}],
                    value=['slider']
                ),
                dcc.Graph(id="graph"),
            ])
        ])
    ])
])

@app.callback(
    Output("graph", "figure"), 
    Input("toggle-rangeslider", "value"))
def display_candlestick(value):
    df_stock = pd.DataFrame(mt.copy_rates_range("EURUSD", mt.TIMEFRAME_D1, datetime(2022, 1, 1), datetime.now()))
    fig = go.Figure(go.Candlestick(
        x=df_stock['time'],
        open=df_stock['open'],
        high=df_stock['high'],
        low=df_stock['low'],
        close=df_stock['close']
    ))

    fig.update_layout(
        xaxis_rangeslider_visible='slider' in value
    )

    return fig

if __name__ == '__main__':
    app.run(debug=True)
