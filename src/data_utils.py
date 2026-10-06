from __future__ import annotations
from pathlib import Path
import pandas as pd
import yfinance as yf

def download_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance."""
    if not ticker.strip():
        raise ValueError("Ticker cannot be empty.")
    df = yf.download(ticker.strip().upper(), period=period, interval="1d",
                     auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No market data returned for ticker: {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    expected = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    result = df[expected].copy()
    result.index = pd.to_datetime(result.index)
    return result.dropna()

def save_data(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)
