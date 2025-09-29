# utils.py
import pandas as pd
import numpy as np
import os
from joblib import dump, load

def ensure_models_dir():
    if not os.path.exists("models"):
        os.makedirs("models")

def save_model(model, name):
    ensure_models_dir()
    path = os.path.join("models", name)
    dump(model, path)
    return path

def load_model(name):
    path = os.path.join("models", name)
    if os.path.exists(path):
        return load(path)
    return None

def infer_datetime_index(df, time_col_candidates=None):
    if time_col_candidates is None:
        time_col_candidates = ["timestamp", "date", "datetime", "time", "ts"]
    for c in df.columns:
        if c.lower() in time_col_candidates:
            try:
                df[c] = pd.to_datetime(df[c])
                df = df.set_index(c).sort_index()
                return df
            except Exception:
                pass
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            df = df.set_index(c).sort_index()
            return df
    try:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    except Exception:
        raise ValueError("Could not detect a datetime column/index. Please provide a timestamp column named one of: timestamp, datetime, date, time, ts.")

def standard_preprocess(df, resample_rule="D", fill_method="ffill"):
    df = df.copy()
    df = infer_datetime_index(df)
    if resample_rule:
        df = df.resample(resample_rule).mean()
    if fill_method:
        if fill_method == "ffill":
            df = df.ffill().bfill()
        elif fill_method == "interpolate":
            df = df.interpolate()
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["year"] = df.index.year
    return df

def prepare_series(df, col):
    s = df[col].dropna()
    return s

def train_test_split_series(series, test_size=0.2):
    n = len(series)
    split = int(n * (1 - test_size))
    train = series.iloc[:split]
    test = series.iloc[split:]
    return train, test
