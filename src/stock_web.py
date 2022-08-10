# coding: utf-8
# stock_web.py

from datetime import datetime

import dash
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

try:
    from components.table import get_prediction_table
    from components.dropdown import *
    from settings import *
    from components.stock_graph import *
except ImportError:
    from src.components.table import get_prediction_table
    from src.components.dropdown import *
    from src.settings import *
    from src.components.stock_graph import *

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
    return f"../models/{predict_type}/{model_type}/{price}.h5"


def get_data(time, price):
    try:
        _time, _date = switch_time(time)
        data = pd.DataFrame(mt.copy_rates_range(price, _time, _date, datetime.now()))
        data["time"] = data['time'].apply(lambda d: datetime.fromtimestamp(d))
        return data
    except NameError:
        data = pd.read_csv("../data/EURUSD_D1.csv")
        data["time"] = data['time'].apply(lambda d: datetime.fromtimestamp(d))
        return data


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
        "💱 Currency Exchange Rate Prediction Analysis Dashboard",
        style={"textAlign": "center", "fontFamily": "monospace"}
    ),

    html.Hr(style={"margin-bottom": "16px"}),

    html.Div([
        get_stock_dropdown(),
        get_time_dropdown(),
        get_algorithm_dropdown(),
    ], style={"display": "flex", "column-gap": "8px", "flexDirection": "row", "marginTop": "8px"}),

    dcc.Loading(
        id="ls-loading-1",
        children=[
            html.H2("📜 Prediction Results", style={"marginTop": "20px", "textAlign": "center"}),
            html.Div(id="prediction-results")
        ],
        type="circle",
    ),

    html.Div([
        html.H2(
            "Line graph",
            style={"textAlign": "center"}
        ),
        *get_actual_graph(df_stock)
    ]),

    html.Div([
        html.H2(
            "Candle graph",
            style={"textAlign": "center"}
        ),
        get_candle_graph(df_stock),
    ]),

    html.Div([
        html.H2(
            "Price of change graph",
            style={"textAlign": "center"}
        ),
        get_price_of_change_graph(df_stock),
    ]),
], style={"padding": "24px 16px"})


# predict data
@app.callback(
    Output("prediction-results", "children"),
    Input('my-interval', 'n_intervals'),  # get data with 1 interval
    Input('time-dropdown', 'value'),
    Input('predict-type-dropdown', 'value'),
    Input('price-dropdown', 'value')
)
def update_prediction_table(_, time, predict_type, price):
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

        return get_prediction_table(closing_price=prediction1, price_of_change=prediction2)

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

    return get_prediction_table(closing_price=prediction[0][0], price_of_change=prediction2[0][0])


# update graph
@app.callback(
    Output("candle-graph", "figure"),
    Output("actual-graph", "figure"),
    Output("price-of-change", "figure"),
    Input('my-interval', 'n_intervals'),  # get data with 1 interval
    Input('time-dropdown', 'value'),
    Input('price-dropdown', 'value')
)
def update_graphs(_, time, price):
    # get data
    data = get_data(time, price)

    return [
        get_candle_graph_figure(data),
        get_actual_graph_figure(data),
        get_price_of_change_graph_figure(data)
    ]


if __name__ == '__main__':
    app.run(debug=True)
