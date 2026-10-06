from __future__ import annotations
import argparse
from sklearn.ensemble import RandomForestRegressor
from data_utils import download_stock_data
from features import FEATURE_COLUMNS, add_features

def main():
    parser = argparse.ArgumentParser(description="Estimate next trading day's closing price.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--period", default="5y")
    args = parser.parse_args()
    raw = download_stock_data(args.ticker, args.period)
    data = add_features(raw)
    split_idx = int(len(data) * 0.8)
    train = data.iloc[:split_idx]
    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(train[FEATURE_COLUMNS], train["Target_Next_Close"])
    latest = data.iloc[[-1]]
    prediction = model.predict(latest[FEATURE_COLUMNS])[0]
    latest_close = latest["Close"].iloc[0]
    print(f"Ticker: {args.ticker.upper()}")
    print(f"Latest close: {latest_close:.2f}")
    print(f"Estimated next-day close: {prediction:.2f}")
    print(f"Estimated change: {prediction - latest_close:+.2f}")
    print(f"Estimated change %: {(prediction / latest_close - 1) * 100:+.2f}%")

if __name__ == "__main__":
    main()
