# models.py
import numpy as np
import pandas as pd
from joblib import dump, load
import os

try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except Exception:
    HAS_PMDARIMA = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    HAS_TF = True
except Exception:
    HAS_TF = False

def arima_train_predict(series, periods=14):
    if not HAS_PMDARIMA:
        raise ImportError("pmdarima not installed. Install pmdarima for ARIMA auto training.")
    model = pm.auto_arima(series, seasonal=False, suppress_warnings=True, error_action='ignore', stepwise=True)
    preds = model.predict(n_periods=periods)
    last_idx = series.index[-1]
    freq = series.index.freq or pd.infer_freq(series.index) or "D"
    idx = pd.date_range(start=last_idx + (series.index[1] - series.index[0]) if len(series)>1 else last_idx + pd.Timedelta(1, unit='D'), periods=periods, freq=freq)
    forecast = pd.Series(preds, index=idx)
    return model, forecast

def prophet_train_predict(series, periods=14):
    if not HAS_PROPHET:
        raise ImportError("prophet not installed. Install prophet to use Prophet model.")
    dfp = series.reset_index()
    dfp.columns = ["ds", "y"]
    model = Prophet()
    model.fit(dfp)
    future = model.make_future_dataframe(periods=periods, freq=series.index.freq or "D")
    forecast = model.predict(future)
    forecast_series = forecast.set_index("ds")["yhat"].iloc[-periods:]
    return model, forecast_series

def xgboost_train_predict(series, exog=None, periods=14):
    if not HAS_XGBOOST:
        raise ImportError("xgboost not installed.")
    df = pd.DataFrame({"y": series})
    lags = 7
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["y"].shift(i)
    if exog is not None:
        exog = exog.reindex(df.index)
        for col in exog.columns:
            df[col] = exog[col]
    df = df.dropna()
    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    X_train = train.drop(columns=["y"])
    y_train = train["y"]
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X_train, y_train)
    last_row = df.drop(columns=["y"]).iloc[-1:].copy()
    preds = []
    for i in range(periods):
        p = model.predict(last_row)[0]
        preds.append(p)
        lag_cols = [c for c in last_row.columns if c.startswith("lag_")]
        lag_cols_sorted = sorted(lag_cols, key=lambda x: int(x.split("_")[1]))
        for j in reversed(range(1, len(lag_cols_sorted))):
            last_row[lag_cols_sorted[j]] = last_row[lag_cols_sorted[j-1]]
        last_row[lag_cols_sorted[0]] = p
    last_idx = series.index[-1]
    freq = series.index.freq or pd.infer_freq(series.index) or "D"
    idx = pd.date_range(start=last_idx + (series.index[1] - series.index[0]) if len(series)>1 else last_idx + pd.Timedelta(1, unit='D'), periods=periods, freq=freq)
    forecast_series = pd.Series(preds, index=idx)
    return model, forecast_series

def lstm_train_predict(series, periods=14, lookback=7, epochs=15, batch_size=8):
    if not HAS_TF:
        raise ImportError("tensorflow not installed. Install tensorflow to use LSTM model.")
    data = series.values.astype("float32")
    mean = data.mean(); std = data.std()
    data_norm = (data - mean) / (std+1e-9)
    X = []
    y = []
    for i in range(lookback, len(data_norm)):
        X.append(data_norm[i - lookback:i])
        y.append(data_norm[i])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    split = int(X.shape[0] * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    model = Sequential()
    model.add(LSTM(32, input_shape=(lookback, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=0)
    last_seq = data_norm[-lookback:].tolist()
    preds = []
    for i in range(periods):
        arr = np.array(last_seq[-lookback:]).reshape((1, lookback, 1))
        p_norm = model.predict(arr, verbose=0)[0][0]
        p = p_norm * std + mean
        preds.append(p)
        last_seq.append(p_norm)
    last_idx = series.index[-1]
    freq = series.index.freq or pd.infer_freq(series.index) or "D"
    idx = pd.date_range(start=last_idx + (series.index[1] - series.index[0]) if len(series)>1 else last_idx + pd.Timedelta(1, unit='D'), periods=periods, freq=freq)
    forecast_series = pd.Series(preds, index=idx)
    return model, forecast_series
