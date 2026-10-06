from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data_utils import download_stock_data
from features import FEATURE_COLUMNS, add_features

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R2": r2_score(y_test, predictions),
    }, predictions

def main():
    parser = argparse.ArgumentParser(description="Train next-day stock close models.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--period", default="5y")
    args = parser.parse_args()
    df = download_stock_data(args.ticker, args.period)
    data = add_features(df)
    split_idx = int(len(data) * 0.8)
    train, test = data.iloc[:split_idx], data.iloc[split_idx:]
    X_train, y_train = train[FEATURE_COLUMNS], train["Target_Next_Close"]
    X_test, y_test = test[FEATURE_COLUMNS], test["Target_Next_Close"]

    models = {
        "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=12,
                                                random_state=42, n_jobs=-1),
    }
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    rows, predictions_by_model = [], {}
    for name, model in models.items():
        metrics, preds = evaluate_model(model, X_train, y_train, X_test, y_test)
        rows.append({"Model": name, **metrics})
        predictions_by_model[name] = preds
    results = pd.DataFrame(rows).sort_values("RMSE")
    print("\nModel evaluation (lower MAE/RMSE is better):")
    print(results.to_string(index=False))
    best_name = results.iloc[0]["Model"]
    comparison = pd.DataFrame(
        {"Actual": y_test.values, "Predicted": predictions_by_model[best_name]},
        index=y_test.index,
    )
    comparison.to_csv(output_dir / "actual_vs_predicted.csv")
    plt.figure(figsize=(12, 6))
    plt.plot(comparison.index, comparison["Actual"], label="Actual")
    plt.plot(comparison.index, comparison["Predicted"], label=f"{best_name} Prediction")
    plt.title(f"{args.ticker.upper()} - Actual vs Predicted Next-Day Close")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
    plt.close()
    print(f"\nBest model: {best_name}")
    print(f"Test observations: {len(test)}")
    print(f"Latest actual close: {df['Close'].iloc[-1]:.2f}")
    print(f"Latest test prediction: {predictions_by_model[best_name][-1]:.2f}")

if __name__ == "__main__":
    main()
