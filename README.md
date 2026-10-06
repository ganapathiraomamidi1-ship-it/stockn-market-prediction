# Stock Market Prediction & Analysis 📈

A practical Python machine-learning project that downloads historical stock data, performs exploratory analysis, engineers technical indicators, trains prediction models, evaluates them on a chronological test set, and estimates the next trading day's closing price.

> Educational use only: This project is not financial advice. Stock-market predictions are uncertain and should not be used alone for investment decisions.

## What the project includes
- Historical OHLCV data with yfinance
- Exploratory analysis of price, returns, volume, moving averages, and volatility
- Technical indicators: SMA, EMA, RSI, daily return, rolling volatility
- Time-series aware train/test split without random shuffling
- Linear Regression and Random Forest comparison
- MAE, RMSE, and R² evaluation
- Actual-vs-predicted visualization
- Next-day prediction CLI
- Jupyter notebook walkthrough

## Tech stack
Python · pandas · NumPy · Matplotlib · scikit-learn · yfinance · Jupyter

## Installation

    git clone https://github.com/ganapathiraomamidi1-ship-it/stockn-market-prediction.git
    cd stockn-market-prediction
    python -m venv .venv
    # Windows: .venv\Scripts\activate
    # macOS/Linux: source .venv/bin/activate
    pip install -r requirements.txt

## Train and evaluate models
    python src/train_model.py --ticker AAPL --period 5y

The program compares Linear Regression and Random Forest, prints MAE/RMSE/R², saves actual-vs-predicted data to outputs/actual_vs_predicted.csv, and saves a chart to outputs/actual_vs_predicted.png.

For Indian stocks, Yahoo Finance symbols can be used, for example:
    python src/train_model.py --ticker RELIANCE.NS --period 5y
    python src/train_model.py --ticker TCS.NS --period 5y

## Generate a next-day estimate
    python src/predict.py --ticker AAPL --period 5y

The exact values depend on the latest data returned by Yahoo Finance when you run the script.

## Methodology
The target variable is the next trading day's closing price: Target(t) = Close(t + 1).

Features are calculated using information available at the current date:
- Open, High, Low, Close, Volume
- 5-day, 20-day, and 50-day SMA
- 12-day and 26-day EMA
- 14-day RSI
- Daily return
- 20-day rolling volatility

The dataset is split chronologically: the first 80% is training data and the most recent 20% is the test set. This avoids leakage that can occur when ordinary random train/test splitting is used on time-series data.

## Models
Linear Regression is used as an interpretable baseline. Random Forest is used as a non-linear ensemble model. The training script selects the model with the lowest test RMSE among the models evaluated.

## Limitations
Price history alone cannot capture every factor that moves markets. This project does not currently model news sentiment, company fundamentals, macroeconomic conditions, transaction costs, slippage, or changing market regimes. A strong historical score does not guarantee future trading performance.

## Future improvements
- Walk-forward validation and rolling retraining
- XGBoost/LightGBM comparison
- LSTM/GRU sequence models
- News and social-media sentiment
- Fundamental and macroeconomic features
- Streamlit dashboard
- Paper-trading/backtesting module with transaction costs
- CI tests and automated model reports

## Author
Ganapathi Rao Mamidi