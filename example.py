import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class QuantAnalyzer:
    """Core analysis class implementing RSI and MACD calculations."""
    
    def __init__(self, ticker: str, start_date: str, end_date: str):
        """Initialize analysis parameters."""
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.fetch_data()
        self.rsi = None
        self.macd = None

    def fetch_data(self) -> pd.DataFrame:
        """Fetch and preprocess market data."""
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        return data['Close']

    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """Calculate the Relative Strength Index (RSI)."""
        delta = self.data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        self.rsi = 100 - (100 / (1 + rs))
        return self.rsi
    
    def calculate_macd(self, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> pd.DataFrame:
        """Calculate the Moving Average Convergence Divergence (MACD)."""
        ema_short = self.data.ewm(span=short_window, adjust=False).mean()
        ema_long = self.data.ewm(span=long_window, adjust=False).mean()
        macd = ema_short - ema_long
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        self.macd = pd.DataFrame({'MACD': macd, 'Signal': signal})
        return self.macd

    def generate_signals(self) -> pd.DataFrame:
        """Generate trading signals based on RSI and MACD."""
        signals = pd.DataFrame(index=self.data.index)
        signals['RSI'] = self.calculate_rsi()
        signals['MACD'] = self.macd['MACD'] if self.macd is not None else self.calculate_macd()['MACD']
        signals['Signal'] = 0
        
        # Buy signal when RSI < 30 and MACD crosses above the signal line
        signals.loc[(signals['RSI'] < 30) & (signals['MACD'] > signals['MACD'].ewm(span=9, adjust=False).mean()), 'Signal'] = 1
        
        # Sell signal when RSI > 70 and MACD crosses below the signal line
        signals.loc[(signals['RSI'] > 70) & (signals['MACD'] < signals['MACD'].ewm(span=9, adjust=False).mean()), 'Signal'] = -1
        
        return signals

    def compute_metrics(self) -> str:
        """Calculate performance metrics."""
        # As an example, here we can count signals
        signals = self.generate_signals()
        long_signals = signals[signals['Signal'] == 1].count()['Signal']
        short_signals = signals[signals['Signal'] == -1].count()['Signal']
        return f'Long Signals: {long_signals}, Short Signals: {short_signals}'

def main():
    """Main execution function."""
    ticker = 'TSLA'
    start_date = '2020-01-01'
    end_date = '2023-10-01'
    
    analyzer = QuantAnalyzer(ticker, start_date, end_date)
    results = analyzer.compute_metrics()
    signals = analyzer.generate_signals()
    
    print("""
    QUANTITATIVE ANALYSIS:
    ---------------------
    Trading Signals Summary:
    ---------------------
    {}
    ---------------------
    ANALYSIS_COMPLETE
    """.format(results))
    
    # Plotting RSI and MACD for visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(analyzer.data.index, analyzer.data.values, label='TSLA Price', color='blue')
    plt.title(f'{ticker} Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    
    plt.subplot(3, 1, 2)
    plt.plot(signals.index, signals['RSI'], label='RSI', color='orange')
    plt.axhline(30, linestyle='--', alpha=0.5, color='red')
    plt.axhline(70, linestyle='--', alpha=0.5, color='green')
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    
    plt.subplot(3, 1, 3)
    plt.plot(signals.index, signals['MACD'], label='MACD', color='purple')
    plt.plot(signals.index, signals['MACD'].ewm(span=9, adjust=False).mean(), label='Signal Line', color='red')
    plt.title('Moving Average Convergence Divergence (MACD)')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()