#!/usr/bin/env python
# coding: utf-8

# In[4]:


import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime
import h5py
import MetaTrader5 as mt

mt.initialize()

login = 5005199849
password = "gyyekbo4"
server = "MetaQuotes-Demo"

mt.login(login, password, server)

account_info = mt.account_info()
print(account_info)

app = dash.Dash()
server = app.server

scaler = MinMaxScaler(feature_range=(0, 1))
df_stock = pd.DataFrame(mt.copy_rates_range("EURUSD",mt.TIMEFRAME_D1,datetime(2022,1,1), datetime.now()))

app.layout = html.Div([

    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),

    dcc.Tabs(id="tabs", children=[

        dcc.Tab(label='NSE-TATAGLOBAL Stock Data', children=[
            html.Div([
                html.H2("Actual closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data": [
                            go.Scatter(
                                x=df_stock['time'],
                                y=df_stock["close"],
                                mode='markers'
                            )
                        ],
                        "layout": go.Layout(
                            title='scatter plot',
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
    app.run_server(debug=True)


# In[ ]:




