from dash import dcc
from dash import html

dropdown_style = {"width": "100%", "marginTop": "16px"}
label_style = {"paddingBottom": "16px"}


def get_stock_dropdown():
    return html.Div([
        html.Label("💱 Currency exchange type", style=label_style),
        dcc.Dropdown(
            options=['EURUSD', 'GBPUSD', 'USDCHF'],
            value='EURUSD',
            id='price-dropdown',
            clearable=False
        ),
    ], style=dropdown_style)


def get_time_dropdown():
    return html.Div([
        html.Label("⌚ Time period", style=label_style),
        dcc.Dropdown(
            ['1 Day', '1 Week', '1 Hour', '1 Minute'],
            '1 Day',
            id='time-dropdown',
            clearable=False
        )
    ], style=dropdown_style)


def get_algorithm_dropdown():
    return html.Div([
        html.Label("🤖 Algorithm", style=label_style),
        dcc.Dropdown(
            options=[
                {"label": "LSTM", "value": "lstm"},
                {"label": "RNN", "value": "rnn"},
                {"label": "XGBOOST", "value": "xgboost"}
            ],
            clearable=False,
            value='lstm',
            id='predict-type-dropdown',
        ),
    ], style=dropdown_style)
