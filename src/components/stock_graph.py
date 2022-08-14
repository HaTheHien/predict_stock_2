import plotly.graph_objects as go
from dash import dcc

def get_candle_graph_figure(data):
    fig = go.Figure([go.Candlestick(
        x=data['time'],
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close']
    )])
    fig.update_layout(xaxis_rangeslider_visible=False, yaxis_title='Candle', xaxis_title='Date time', )
    return fig


def get_actual_graph_figure(data):
    return {
        "data": [
            go.Scatter(
                x=data['time'],
                y=data["close"],
                mode="lines",
            )
        ],
        "layout": go.Layout(
            xaxis={'title': 'Date time'},
            yaxis={'title': 'Closing Rate'}
        )
    }


def get_price_of_change_graph_figure(data):
    return {
        "data": [
            go.Scatter(
                x=data['time'],
                y=data["close"] - data["open"],
                mode="lines",
            )
        ],
        "layout": go.Layout(
            xaxis={'title': 'Date time'},
            yaxis={'title': 'Rate of change'}
        )
    }


def get_relative_strength_index_figure(data):
    return {
        "data": [
            go.Scatter(
                x=data['time'],
                y=data["rsi"],
                mode="lines",
            )
        ],
        "layout": go.Layout(
            xaxis={'title': 'Timestamp'},
            yaxis={'title': 'Relative Strength Index'}
        )
    }

def get_moving_average_figure(data):
    return {
        "data": [
            go.Scatter(
                x=data['time'],
                y=data["ma"],
                mode="lines",
            )
        ],
        "layout": go.Layout(
            xaxis={'title': 'Timestamp'},
            yaxis={'title': 'Moving Average'}
        )
    }

def get_actual_graph(df_stock):
    return dcc.Graph(
        id="actual-graph",
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


def get_candle_graph(df_stock):
    return dcc.Graph(
        id="candle-graph",
        figure=go.Figure(go.Candlestick(
            x=df_stock['time'],
            open=df_stock['open'],
            high=df_stock['high'],
            low=df_stock['low'],
            close=df_stock['close']
        )))


def get_price_of_change_graph(df_stock):
    return dcc.Graph(
        id="price-of-change",
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
    )

def get_relative_strength_index_graph(df_stock):
    #data=rsi(data)
    return dcc.Graph(
        id="relative-strength-index",
        figure={
            "data": [
                go.Scatter(
                    x=df_stock['time'],
                    y=df_stock["rsi"],
                    mode="lines",
                )
            ],
            "layout": go.Layout(
                xaxis={'title': 'Timestamp'},
                yaxis={'title': 'Relative Strength Index'}
            )
        }
    )

def get_moving_average_graph(df_stock):
    #data=moving_average(data)
    return dcc.Graph(
            id="moving-average",
            figure={
                "data": [
                    go.Scatter(
                        x=df_stock['time'],
                        y=df_stock["ma"],
                        mode="lines",
                    )
                ],
                "layout": go.Layout(
                    xaxis={'title': 'Timestamp'},
                    yaxis={'title': 'Moving Average'}
                )
            }
        )