# Analysis generated on 2024-12-30 22:07:15
# Task: Backtest a momentum strategy on SPY over 2 years with daily rebalancing and performance metrics.

import os
import pandas as pd
import numpy as np
import yfinance as yf
import vectorbt as vbt
import optopsy as op
from datetime import datetime, timedelta
import quantstats as qs
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# Create output directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

import yfinance as yf
import vectorbt as vbt
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple

logging.basicConfig(level=logging.DEBUG)

def fetch_data(ticker: str, start: str, end: str) -> pd.Series:
    if not isinstance(ticker, str) or not ticker.isalpha() or len(ticker) > 5:
        logging.error("Invalid ticker format.")
        raise ValueError("Ticker must be a non-empty string consisting of alphabetic characters, with a maximum length of 5.")
    
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            logging.error("Fetched data is empty.")
            raise ValueError("No data retrieved.")
        logging.info("Data fetched successfully for ticker: %s", ticker)
        return data['Close']
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        raise

def calculate_signals(prices: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series]:
    momentum = prices.pct_change(periods=window)
    entries = momentum > 0
    exits = momentum <= 0
    return entries, exits

def backtest_momentum_strategy(ticker: str, start: str, end: str) -> Dict[str, Any]:
    prices = fetch_data(ticker, start, end)
    entries, exits = calculate_signals(prices)

    portfolio = vbt.Portfolio.from_signals(prices, entries, exits, freq='1D')
    
    metrics = {
        "Total Return": portfolio.total_return(),
        "Annualized Return": portfolio.annualized_return(),
        "Max Drawdown": portfolio.max_drawdown(),
        "Sharpe Ratio": portfolio.sharpe_ratio(),
        "Trades": portfolio.trades,
        "Winning Trades": portfolio.winning_trades,
        "Losing Trades": portfolio.losing_trades,
    }
    
    return metrics

if __name__ == "__main__":
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    try:
        results = backtest_momentum_strategy("SPY", start_date, end_date)
        print(results)
    except Exception as e:
        logging.error(f"Backtesting failed: {e}")

# Save results
if 'results_df' in locals():
    results_df.to_csv(f'outputs/results_20241230_220715.csv')
if 'stats_df' in locals():
    stats_df.to_csv(f'outputs/stats_20241230_220715.csv')
if 'performance_df' in locals():
    performance_df.to_csv(f'outputs/performance_20241230_220715.csv')
