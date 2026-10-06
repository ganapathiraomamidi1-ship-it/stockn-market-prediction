from __future__ import annotations
import numpy as np
import pandas as pd

def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["SMA_5"] = data["Close"].rolling(5).mean()
    data["SMA_20"] = data["Close"].rolling(20).mean()
    data["SMA_50"] = data["Close"].rolling(50).mean()
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()
    data["RSI_14"] = calculate_rsi(data["Close"], 14)
    data["Daily_Return"] = data["Close"].pct_change()
    data["Volatility_20"] = data["Daily_Return"].rolling(20).std()
    data["Target_Next_Close"] = data["Close"].shift(-1)
    return data.dropna().copy()

FEATURE_COLUMNS = [
    "Open","High","Low","Close","Volume","SMA_5","SMA_20","SMA_50",
    "EMA_12","EMA_26","RSI_14","Daily_Return","Volatility_20"
]
