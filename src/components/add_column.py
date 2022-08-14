def relative_strength_index(df,periods=14,ema=True):
    close_delta = df['close'].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema == True:
	    # Use exponential moving average
        ma_up = up.ewm(com = periods, adjust=False).mean()
        ma_down = down.ewm(com = periods, adjust=False).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = periods, adjust=False).mean()
        ma_down = down.rolling(window = periods, adjust=False).mean()
        
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    
    tmp=df.copy()

    tmp["rsi"]=rsi
    return tmp

def moving_average(df,period=20):
    #df["ma"]=df["close"].rolling(period).mean()
    tmp=df.copy()
    tmp["ma"]=tmp["close"].ewm(span=period,adjust=False).mean()
    return tmp