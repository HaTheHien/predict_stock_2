#!/usr/bin/env python
# coding: utf-8

from dash import dcc, html, Dash
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime
from src.settings import LOGIN, PASSWORD, SERVER

print(LOGIN, PASSWORD, SERVER)
try:
    import MetaTrader5 as mt

    mt.initialize()
    login = LOGIN
    password = PASSWORD
    server = SERVER

    mt.login(login, password, server)

    account_info = mt.account_info()
    print(account_info)
    df_stock = pd.DataFrame(mt.copy_rates_range("EURUSD", mt.TIMEFRAME_D1, datetime(2022, 1, 1), datetime.now()))
except ImportError:
    df_stock = pd.read_csv("../data/EURUSD_D1.csv")

app = Dash()
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
                            title='scatter line',
                            xaxis={'title': 'Timestamp'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                ),
            ])
        ]),
    ])
])

if __name__ == '__main__':
    app.run(debug=True)
    pass
