# Analysis generated on 2024-12-30 00:45:29
# Task: Calculate technical indicators (RSI, MACD) for TSLA and identify potential trading signals.

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
import numpy as np
import logging
from typing import Tuple, Optional, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_data(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            raise ValueError("No data fetched for the given ticker.")
        return data
    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        return None
    except Exception as e:
        logging.error(f"Exception while fetching data for {ticker}: {e}")
        return None

def calculate_indicators(data: pd.DataFrame) -> Optional[Dict[str, pd.Series]]:
    try:
        if len(data) < 26:
            raise ValueError("Not enough data to calculate MACD.")
        close = data['Close']
        rsi = vbt.RSI.run(close, window=14).rsi
        macd = vbt.MACD.run(close, short_window=12, long_window=26, signal_window=9)
        return {
            'rsi': rsi,
            'macd': macd.macd,
            'macd_signal': macd.signal
        }
    except KeyError as ke:
        logging.error(f"KeyError: {ke} - Check the column names in the DataFrame.")
        return None
    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        return None
    except Exception as e:
        logging.error(f"Exception while calculating indicators: {e}")
        return None

def identify_signals(indicators: Dict[str, pd.Series], prices: pd.Series) -> Optional[pd.DataFrame]:
    try:
        buy_signals = (indicators['macd'] > indicators['macd_signal']) & (indicators['rsi'] < 30)
        sell_signals = (indicators['macd'] < indicators['macd_signal']) & (indicators['rsi'] > 70)
        signals_df = pd.DataFrame({
            'buy_signal': buy_signals,
            'sell_signal': sell_signals,
            'date': prices.index,
            'price': prices.values
        })
        return signals_df[(signals_df['buy_signal'] | signals_df['sell_signal'])]
    except Exception as e:
        logging.error(f"Exception while identifying signals: {e}")
        return None

def main(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    data = fetch_data(ticker, start, end)
    if data is None:
        return None

    indicators = calculate_indicators(data)
    if indicators is None:
        return None

    signals = identify_signals(indicators, data['Close'])
    return signals

results = main('TSLA', '2020-01-01', '2023-10-01')

# Save results
if 'results_df' in locals():
    results_df.to_csv(f'outputs/results_20241230_004529.csv')
if 'stats_df' in locals():
    stats_df.to_csv(f'outputs/stats_20241230_004529.csv')
if 'performance_df' in locals():
    performance_df.to_csv(f'outputs/performance_20241230_004529.csv')
