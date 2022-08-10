import dash


def get_prediction_table(closing_price, price_of_change):
    return dash.dash_table.DataTable(
        columns=[
            {"id": "Criteria", "name": "Criterion"},
            {"id": "Predict", "name": "Prediction"}
        ],
        data=[
            {"Criteria": "Closing", "Predict": closing_price},
            {"Criteria": "Price of Change", "Predict": price_of_change}
        ],
        style_header={'textAlign': 'center', "fontSize": "20px"},
        style_cell={'textAlign': 'center', "fontSize": "14px"},
        style_table={"marginBottom": "50px"},
    )
