# coding: utf-8
# stock_web.py

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import numpy as np
from src.settings import LOGIN, PASSWORD, SERVER
import plotly.graph_objects as go
from xgboost import XGBRegressor, Booster

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
    df_stock = pd.read_csv("./data/EURUSD_D1.csv")

app = dash.Dash()
server = app.server
scaler = MinMaxScaler(feature_range=(0, 1))


def switch_time(time):
    if time == "1 Week":
        return mt.TIMEFRAME_W1, datetime(2022, 1, 1)
    if time == "1 Hour":
        return mt.TIMEFRAME_H1, datetime(2022, 1, 1)
    if time == "1 Minute":
        return mt.TIMEFRAME_M1, datetime(2022, 7, 20)
    return mt.TIMEFRAME_D1, datetime(2022, 1, 1)


# modelType: LSTM, RNN; predictType: Closing
def get_path_model(predict_type, model_type, price):
    return f"./models/{predict_type}/{model_type}/{price}.h5"


def get_data(time, price):
    try:
        _time, _date = switch_time(time)
        data = pd.DataFrame(mt.copy_rates_range(price, _time, _date, datetime.now()))
        data["time"] = data['time'].apply(lambda d: datetime.fromtimestamp(d))
        return data
    except NameError:
        data = pd.read_csv("./data/EURUSD_D1.csv")
        data["time"] = data['time'].apply(lambda d: datetime.fromtimestamp(d))
        return data


def get_graph_candle(data):
    fig = go.Figure([go.Candlestick(
        x=data['time'],
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close']
    )])
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig


def get_actual_graph(data):
    return {
        "data": [
            go.Scatter(
                x=data['time'],
                y=data["close"],
                mode="lines",
            )
        ],
        "layout": go.Layout(
            xaxis={'title': 'Timestamp'},
            yaxis={'title': 'Closing Rate'}
        )
    }


def get_price_of_change_graph(data):
    return {
        "data": [
            go.Scatter(
                x=data['time'],
                y=data["close"] - data["open"],
                mode="lines",
            )
        ],
        "layout": go.Layout(
            xaxis={'title': 'Timestamp'},
            yaxis={'title': 'Rate of change'}
        )
    }


# function xgboost
def roc(data, n):
    N = data["close"].diff(n)
    M = data["close"].shift(n)
    roc = pd.Series(N / M, name="roc")
    data = data.join(roc)
    return data


def preprocess_roc(data):
    data_close = data[["close"]].copy()
    tmp = roc(data_close, 5)
    data_roc = tmp[["roc"]].copy()
    data_roc["target"] = data_roc.roc.shift(-1)
    data_roc.dropna(inplace=True)
    return data_roc


def predict_value(data, percent, path=""):
    # Load model
    model = XGBRegressor(objective="reg:squarederror", n_estimators=1000)
    model.load_model(path)

    # get test
    test = data.values[-1:]

    # Predict value
    predictions = []
    for i in range(len(test)):
        val = np.array(test[i, 0]).reshape(1, -1)
        pred = model.predict(val)
        predictions.append(pred[0])
    return predictions[0]


app.layout = html.Div([
    dcc.Interval(id='my-interval', interval=60000),  # update html in milliseconds (cur 1 minute load again)

    html.H1(
        "Currency Exchange Rate Prediction Analysis Dashboard",
        style={"textAlign": "center"}
    ),

    html.Div([
        dcc.Dropdown(
            options=['EURUSD', 'GBPUSD', 'USDCHF'],
            value='EURUSD',
            id='price-dropdown'
        ),
        dcc.Dropdown(['1 Day', '1 Week', '1 Hour', '1 Minute'], '1 Day', id='time-dropdown'),
        dcc.Dropdown(
            options=[
                {"label": "LSTM", "value": "lstm"},
                {"label": "RNN", "value": "rnn"},
                {"label": "XGBOOST", "value": "xgboost"}
            ],
            value='lstm',
            id='predict-type-dropdown'
        ),
    ]),

    dcc.Loading(
        id="ls-loading-1",
        style={"height": "100px"},
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

    html.Div([
        html.H2(
            "Line graph",
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
                # "hover_data": {"time": "|%B %d, %Y"},
                "layout": go.Layout(
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

    html.Div([
        html.H2(
            "Price of change graph",
            style={"textAlign": "center"}
        ),
        dcc.Graph(
            id="Price of change",
            figure={
                "data": [
                    go.Scatter(
                        x=df_stock['time'],
                        y=df_stock["close"] - df_stock["open"],
                        mode="lines",
                    )
                ],
                "layout": go.Layout(
                    xaxis={'title': 'Timestamp'},
                    yaxis={'title': 'Closing Rate'}
                )
            }
        ),
    ]),
])


# predict data
@app.callback(
    Output("predictClosing", "children"),
    Output("predictPriceOfChange", "children"),
    Input('my-interval', 'n_intervals'),  # get data with 1 interval
    Input('time-dropdown', 'value'),
    Input('predict-type-dropdown', 'value'),
    Input('price-dropdown', 'value')
)
def multi_output(n_intervals, time, predict_type, price):
    prediction_days = 10

    # get data
    data = get_data(time, price)

    if predict_type == "xgboost":
        # init
        percent = 0.2

        # copy data closing
        data_close = data[["close"]].copy()
        data_close["target"] = data_close.close.shift(-1)
        data_close.dropna(inplace=True)
        prediction1 = predict_value(data_close, percent, get_path_model("closing", predict_type, price))

        #
        data_roc1 = preprocess_roc(data)
        prediction2 = predict_value(data_close, percent, get_path_model("price_of_change", predict_type, price))

        return ["Predict closing: " + str(prediction1), "Predict price of change: " + str(prediction2)]

    # closing model
    scaled_data_closing = scaler.fit_transform(data['close'].values[-22:].reshape(-1, 1))
    model_closing = load_model(get_path_model("closing", predict_type, price))

    x_train = []

    for x in range(len(scaled_data_closing) - 12, len(scaled_data_closing)):
        x_train.append(scaled_data_closing[x - prediction_days:x, 0])

    x_train = np.array(x_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    real_data = [x_train[len(x_train + 1) - prediction_days:len(x_train), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model_closing.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(prediction)

    # Price of change
    scaled_data_price_of_change = scaler.fit_transform((data['close'] - data['open']).values[-22:].reshape(-1, 1))
    model_price_of_change = load_model(get_path_model("price_of_change", predict_type, price))

    x_train = []

    for x in range(len(scaled_data_price_of_change) - 12, len(scaled_data_price_of_change)):
        x_train.append(scaled_data_price_of_change[x - prediction_days:x, 0])

    x_train = np.array(x_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    real_data = [x_train[len(x_train + 1) - prediction_days:len(x_train), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction2 = model_price_of_change.predict(real_data)
    prediction2 = scaler.inverse_transform(prediction2)
    print(prediction2)

    return [f"Predict closing: {str(prediction[0][0])}", f"Predict price of change: {str(prediction2[0][0])}"]


# update graph
@app.callback(
    Output("Candle graph", "figure"),
    Output("Actual Data", "figure"),
    Output("Price of change", "figure"),
    Input('my-interval', 'n_intervals'),  # get data with 1 interval
    Input('time-dropdown', 'value'),
    Input('price-dropdown', 'value')
)
def multi_output(n_intervals, time, price):
    # get data
    data = get_data(time, price)

    return [get_graph_candle(data), get_actual_graph(data), get_price_of_change_graph(data)]


if __name__ == '__main__':
    app.run(debug=True)
