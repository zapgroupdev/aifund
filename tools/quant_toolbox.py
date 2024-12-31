import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default plot settings
WORKSPACE_DIR = 'workspace'
os.makedirs(WORKSPACE_DIR, exist_ok=True)

plt.style.use('fivethirtyeight')
DEFAULT_FIGSIZE = (12, 6)
DEFAULT_DPI = 100
DEFAULT_COLORS = {
    'price': '#2E86C1',
    'volume': '#85929E',
    'up': '#2ECC71',
    'down': '#E74C3C',
    'ma': ['#E67E22', '#8E44AD', '#16A085'],
    'signal': ['#F1C40F', '#CB4335']
}

async def fetch_market_data(
    symbol: str,
    source: str = 'yahoo',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeframe: str = '1d',
    include_adjusted: bool = True
) -> pd.DataFrame:
    """Fetch market data from various sources with consistent output format.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
        source: Data source to use ('yahoo', 'alpaca')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        timeframe: Data timeframe ('1d', '1h', '15min', etc.)
        include_adjusted: Whether to include adjusted prices
    
    Returns:
        DataFrame with columns: [Open, High, Low, Close, Volume, (Adj Close)]
        Index is DatetimeIndex in UTC
    """
    try:
        # Parameter validation and defaults
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if not start_date:
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Fetch data based on source
        if source.lower() == 'yahoo':
            df = await _fetch_yahoo_data(symbol, start_date, end_date, timeframe, include_adjusted)
        
        elif source.lower() == 'alpaca':
            df = await _fetch_alpaca_data(symbol, start_date, end_date, timeframe)
        
        else:
            raise ValueError(f"Unsupported data source: {source}")
        
        if df.empty:
            raise ValueError(f"No data found for {symbol}")
        
        # Ensure consistent format
        df = _standardize_dataframe(df, include_adjusted)
        logger.info(f"Successfully fetched {len(df)} rows of {timeframe} data for {symbol}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        raise

async def _fetch_yahoo_data(symbol: str, start_date: str, end_date: str, 
                          timeframe: str, include_adjusted: bool) -> pd.DataFrame:
    """Fetch data from Yahoo Finance."""
    try:
        # Map timeframe to yfinance interval
        interval_map = {
            '1d': '1d',
            '1h': '1h',
            '15min': '15m',
            '5min': '5m',
            '1min': '1m'
        }
        interval = interval_map.get(timeframe, '1d')
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        
        return df
    
    except Exception as e:
        raise ValueError(f"Yahoo Finance error: {str(e)}")

async def _fetch_alpaca_data(symbol: str, start_date: str, end_date: str, 
                           timeframe: str) -> pd.DataFrame:
    """Fetch data from Alpaca Markets."""
    try:
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not api_secret:
            raise ValueError("Alpaca API credentials not found")
        
        # Map timeframe to Alpaca TimeFrame
        timeframe_map = {
            '1d': TimeFrame.Day,
            '1h': TimeFrame.Hour,
            '15min': TimeFrame(15, 'Min'),
            '5min': TimeFrame(5, 'Min'),
            '1min': TimeFrame.Minute
        }
        tf = timeframe_map.get(timeframe, TimeFrame.Day)
        
        client = StockHistoricalDataClient(api_key, api_secret)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=datetime.strptime(start_date, '%Y-%m-%d'),
            end=datetime.strptime(end_date, '%Y-%m-%d')
        )
        
        response = client.get_stock_bars(request)
        df = pd.DataFrame([bar for bar in response[symbol]])
        
        return df
    
    except Exception as e:
        raise ValueError(f"Alpaca Markets error: {str(e)}")

def _standardize_dataframe(df: pd.DataFrame, include_adjusted: bool) -> pd.DataFrame:
    """Standardize DataFrame format across different sources."""
    # Ensure UTC timezone
    if df.index.tz is not None:
        df.index = df.index.tz_convert('UTC')
    else:
        df.index = df.index.tz_localize('UTC')
    
    # Standardize column names
    column_map = {
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume',
        'Adj Close': 'Adj Close'
    }
    df = df.rename(columns={k.lower(): v for k, v in column_map.items()})
    
    # Select and order columns
    columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if include_adjusted and 'Adj Close' in df.columns:
        columns.append('Adj Close')
    
    return df[columns]

async def calculate_technical_indicators(
    data: pd.DataFrame,
    indicators: List[str],
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Union[pd.Series, Dict[str, pd.Series]]]:
    """Calculate technical indicators for the given data.
    
    Args:
        data: OHLCV DataFrame
        indicators: List of indicators to calculate ('sma', 'ema', 'rsi', 'macd', 'bbands')
        params: Parameters for each indicator
            Example: {
                'sma': {'window': 20},
                'ema': {'window': 20},
                'rsi': {'window': 14},
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'bbands': {'window': 20, 'num_std': 2}
            }
    
    Returns:
        Dictionary of calculated indicators
    """
    if not params:
        params = {}
    
    results = {}
    
    for indicator in indicators:
        if indicator == 'sma':
            window = params.get('sma', {}).get('window', 20)
            results['SMA'] = data['Close'].rolling(window=window).mean()
        
        elif indicator == 'ema':
            window = params.get('ema', {}).get('window', 20)
            results['EMA'] = data['Close'].ewm(span=window, adjust=False).mean()
        
        elif indicator == 'rsi':
            window = params.get('rsi', {}).get('window', 14)
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            results['RSI'] = 100 - (100 / (1 + rs))
        
        elif indicator == 'macd':
            fast = params.get('macd', {}).get('fast', 12)
            slow = params.get('macd', {}).get('slow', 26)
            signal = params.get('macd', {}).get('signal', 9)
            
            fast_ema = data['Close'].ewm(span=fast, adjust=False).mean()
            slow_ema = data['Close'].ewm(span=slow, adjust=False).mean()
            macd_line = fast_ema - slow_ema
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            
            results['MACD'] = {
                'MACD': macd_line,
                'Signal': signal_line,
                'Histogram': macd_line - signal_line
            }
        
        elif indicator == 'bbands':
            window = params.get('bbands', {}).get('window', 20)
            num_std = params.get('bbands', {}).get('num_std', 2)
            
            sma = data['Close'].rolling(window=window).mean()
            std = data['Close'].rolling(window=window).std()
            
            results['BBands'] = {
                'Middle': sma,
                'Upper': sma + (std * num_std),
                'Lower': sma - (std * num_std)
            }
    
    return results

def _generate_line_plot(data: pd.DataFrame, **kwargs):
    """Generate a line plot."""
    columns = kwargs.get('columns', [data.columns[0]])
    colors = kwargs.get('colors', {})
    
    for i, column in enumerate(columns):
        color = colors.get(column, DEFAULT_COLORS['ma'][i % len(DEFAULT_COLORS['ma'])])
        plt.plot(data.index, data[column], label=column, color=color)

def _generate_scatter_plot(data: pd.DataFrame, **kwargs):
    """Generate a scatter plot."""
    x_col = kwargs['x']
    y_col = kwargs['y']
    color = kwargs.get('color', DEFAULT_COLORS['signal'][0])
    label = kwargs.get('label', '')
    
    plt.scatter(data[x_col], data[y_col], label=label, color=color)

def _generate_candlestick_plot(data: pd.DataFrame, **kwargs):
    """Generate a candlestick plot."""
    df = data.copy()
    if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        raise ValueError("Data must contain OHLC columns")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=kwargs.get('figsize', DEFAULT_FIGSIZE),
                                  gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    mpf.plot(df, type='candle', style='charles',
            ax=ax1, volume=False,
            colorup=DEFAULT_COLORS['up'],
            colordown=DEFAULT_COLORS['down'])
    
    if kwargs.get('volume', True) and 'Volume' in df.columns:
        ax2.bar(df.index, df['Volume'], color=DEFAULT_COLORS['volume'], alpha=0.5)
        ax2.set_ylabel('Volume')
    
    ax1.set_ylabel('Price')
    plt.xticks(rotation=45)

def _generate_technical_plot(data: pd.DataFrame, **kwargs):
    """Generate a technical analysis plot with indicators."""
    indicators = kwargs.get('indicators', {})
    n_subplots = 1 + len(indicators)
    
    fig, axes = plt.subplots(n_subplots, 1, figsize=kwargs.get('figsize', (12, 6*n_subplots)),
                            sharex=True)
    if n_subplots == 1:
        axes = [axes]
    
    axes[0].plot(data.index, data['Close'], label='Price', color=DEFAULT_COLORS['price'])
    
    for i, (name, indicator_data) in enumerate(indicators.items(), 1):
        if isinstance(indicator_data, pd.Series):
            axes[i].plot(data.index, indicator_data, label=name,
                       color=DEFAULT_COLORS['signal'][0])
        elif isinstance(indicator_data, dict):
            for j, (line_name, line_data) in enumerate(indicator_data.items()):
                axes[i].plot(data.index, line_data, label=f"{name} {line_name}",
                           color=DEFAULT_COLORS['ma'][j % len(DEFAULT_COLORS['ma'])])
        axes[i].set_ylabel(name)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    axes[0].set_ylabel('Price')
    axes[-1].set_xlabel('Date')

async def generate_plot(
    data: pd.DataFrame,
    chart_type: str,
    title: str,
    overlays: Optional[List[Dict[str, Any]]] = None,
    subplots: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[str] = None,
    **kwargs: Any
) -> str:
    """Generate an interactive financial chart using Plotly.
    
    Args:
        data: OHLCV DataFrame
        chart_type: Type of main chart ('candlestick', 'line', 'ohlc')
        title: Chart title
        overlays: List of overlays to add to main chart
        subplots: List of subplots to add
        output_path: Path to save HTML file
        **kwargs: Additional styling parameters
    
    Returns:
        str: Path to saved HTML file
    """
    try:
        # Determine number of subplots
        n_subplots = 1 + (len(subplots) if subplots else 0)
        subplot_heights = _calculate_subplot_heights(n_subplots)
        
        # Create figure with subplots
        fig = make_subplots(
            rows=n_subplots,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=subplot_heights
        )
        
        # Add main chart
        if chart_type == 'candlestick':
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
        elif chart_type == 'ohlc':
            fig.add_trace(
                go.Ohlc(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
        elif chart_type == 'line':
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
        
        # Add overlays to main chart
        if overlays:
            for overlay in overlays:
                _add_overlay(fig, data, overlay, row=1)
        
        # Add subplots
        if subplots:
            for i, subplot in enumerate(subplots, start=2):
                _add_subplot(fig, data, subplot, row=i)
        
        # Update layout
        fig.update_layout(
            title=title,
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            height=150 * n_subplots,
            template='plotly_dark' if kwargs.get('dark_mode', True) else 'plotly_white',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Save plot
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f'{WORKSPACE_DIR}/chart_{timestamp}.html'
        
        pio.write_html(fig, output_path)
        logger.info(f"Chart saved to {output_path}")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating plot: {str(e)}")
        raise

def _calculate_subplot_heights(n_subplots: int) -> List[float]:
    """Calculate relative heights for subplots."""
    if n_subplots == 1:
        return [1.0]
    elif n_subplots == 2:
        return [0.7, 0.3]
    else:
        # Main chart gets 50%, rest split evenly
        subplot_height = 0.5 / (n_subplots - 1)
        return [0.5] + [subplot_height] * (n_subplots - 1)

def _add_overlay(fig: go.Figure, data: pd.DataFrame, overlay: Dict[str, Any], row: int):
    """Add overlay to the main chart."""
    if overlay['type'] == 'sma':
        sma = data['Close'].rolling(window=overlay.get('length', 20)).mean()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=sma,
                mode='lines',
                name=f"SMA({overlay.get('length', 20)})",
                line=dict(color=overlay.get('color', 'blue'))
            ),
            row=row, col=1
        )
    
    elif overlay['type'] == 'ema':
        ema = data['Close'].ewm(span=overlay.get('length', 20), adjust=False).mean()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=ema,
                mode='lines',
                name=f"EMA({overlay.get('length', 20)})",
                line=dict(color=overlay.get('color', 'orange'))
            ),
            row=row, col=1
        )
    
    elif overlay['type'] == 'bbands':
        length = overlay.get('length', 20)
        std = overlay.get('std', 2)
        sma = data['Close'].rolling(window=length).mean()
        std_dev = data['Close'].rolling(window=length).std()
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=sma + (std_dev * std),
                mode='lines',
                name=f'Upper BB({length})',
                line=dict(color='gray', dash='dash')
            ),
            row=row, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=sma - (std_dev * std),
                mode='lines',
                name=f'Lower BB({length})',
                line=dict(color='gray', dash='dash'),
                fill='tonexty'
            ),
            row=row, col=1
        )

def _add_subplot(fig: go.Figure, data: pd.DataFrame, subplot: Dict[str, Any], row: int):
    """Add subplot below main chart."""
    if subplot['type'] == 'volume':
        colors = np.where(data['Close'] >= data['Open'], 'green', 'red')
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors
            ),
            row=row, col=1
        )
        fig.update_yaxes(title_text="Volume", row=row, col=1)
    
    elif subplot['type'] == 'rsi':
        length = subplot.get('length', 14)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=rsi,
                mode='lines',
                name=f'RSI({length})'
            ),
            row=row, col=1
        )
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=row, col=1)
        fig.update_yaxes(title_text="RSI", row=row, col=1)

async def calculate_performance_metrics(
    data: pd.DataFrame,
    returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """Calculate performance metrics for a trading strategy.
    
    Args:
        data: DataFrame with price data (must have 'Close' column if returns not provided)
        returns: Optional pre-calculated returns series
        risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
    
    Returns:
        Dictionary of performance metrics
    """
    if returns is None:
        returns = data['Close'].pct_change()
    
    # Convert annual risk-free rate to match returns frequency
    rf_daily = (1 + risk_free_rate) ** (1/252) - 1
    excess_returns = returns - rf_daily
    
    # Calculate metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252/len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std()
    sortino_ratio = np.sqrt(252) * excess_returns.mean() / returns[returns < 0].std()
    max_drawdown = (data['Close'] / data['Close'].expanding(min_periods=1).max() - 1).min()
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown
    }

async def calculate_metrics(
    data: pd.DataFrame,
    metrics: List[str],
    settings: Optional[Dict[str, Any]] = None,
    returns: Optional[pd.Series] = None,
    benchmark: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Calculate financial and statistical metrics.
    
    Args:
        data: OHLCV DataFrame
        metrics: List of metrics to calculate
            Supported metrics:
            - returns: ['total', 'annual', 'monthly', 'daily']
            - risk: ['volatility', 'var', 'cvar', 'beta']
            - ratios: ['sharpe', 'sortino', 'calmar', 'information']
            - drawdown: ['max_drawdown', 'avg_drawdown', 'drawdown_duration']
            - statistics: ['skew', 'kurtosis', 'autocorr']
        settings: Configuration for calculations
        returns: Optional pre-calculated returns
        benchmark: Optional benchmark data for relative metrics
    
    Returns:
        Dictionary of calculated metrics with clear labels
    """
    try:
        # Default settings
        default_settings = {
            'risk_free_rate': 0.02,
            'periods_per_year': 252,
            'var_confidence': 0.95,
            'rolling_window': 20
        }
        settings = {**default_settings, **(settings or {})}
        
        # Calculate returns if not provided
        if returns is None:
            returns = data['Close'].pct_change()
        
        # Initialize results dictionary
        results = {}
        
        # Calculate requested metrics
        for metric in metrics:
            if metric in ['total', 'annual', 'monthly', 'daily']:
                results.update(_calculate_return_metrics(returns, settings))
            
            elif metric in ['volatility', 'var', 'cvar', 'beta']:
                results.update(_calculate_risk_metrics(returns, settings, benchmark))
            
            elif metric in ['sharpe', 'sortino', 'calmar', 'information']:
                results.update(_calculate_ratio_metrics(returns, settings, benchmark))
            
            elif metric in ['max_drawdown', 'avg_drawdown', 'drawdown_duration']:
                results.update(_calculate_drawdown_metrics(data['Close'], returns))
            
            elif metric in ['skew', 'kurtosis', 'autocorr']:
                results.update(_calculate_statistical_metrics(returns, settings))
        
        return results
    
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise

def _calculate_return_metrics(returns: pd.Series, settings: Dict[str, Any]) -> Dict[str, float]:
    """Calculate return-based metrics."""
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (settings['periods_per_year']/len(returns)) - 1
    
    # Monthly returns
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    avg_monthly_return = monthly_returns.mean()
    
    # Daily statistics
    avg_daily_return = returns.mean()
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'avg_monthly_return': avg_monthly_return,
        'avg_daily_return': avg_daily_return
    }

def _calculate_risk_metrics(
    returns: pd.Series, 
    settings: Dict[str, Any],
    benchmark: Optional[pd.DataFrame]
) -> Dict[str, float]:
    """Calculate risk metrics."""
    # Annualized volatility
    volatility = returns.std() * np.sqrt(settings['periods_per_year'])
    
    # Value at Risk
    var = returns.quantile(1 - settings['var_confidence'])
    
    # Conditional VaR (Expected Shortfall)
    cvar = returns[returns <= var].mean()
    
    # Beta (if benchmark provided)
    beta = None
    if benchmark is not None and 'Close' in benchmark:
        benchmark_returns = benchmark['Close'].pct_change()
        covariance = returns.cov(benchmark_returns)
        variance = benchmark_returns.var()
        beta = covariance / variance
    
    metrics = {
        'volatility': volatility,
        'value_at_risk': var,
        'conditional_var': cvar
    }
    
    if beta is not None:
        metrics['beta'] = beta
    
    return metrics

def _calculate_ratio_metrics(
    returns: pd.Series, 
    settings: Dict[str, Any],
    benchmark: Optional[pd.DataFrame]
) -> Dict[str, float]:
    """Calculate ratio metrics."""
    # Convert annual risk-free rate to match returns frequency
    rf_rate = (1 + settings['risk_free_rate']) ** (1/settings['periods_per_year']) - 1
    excess_returns = returns - rf_rate
    
    # Sharpe Ratio
    sharpe = np.sqrt(settings['periods_per_year']) * excess_returns.mean() / returns.std()
    
    # Sortino Ratio
    downside_returns = returns[returns < 0]
    sortino = np.sqrt(settings['periods_per_year']) * excess_returns.mean() / downside_returns.std()
    
    # Calmar Ratio
    max_dd = _calculate_drawdown_metrics(None, returns)['max_drawdown']
    calmar = abs(returns.mean() * settings['periods_per_year'] / max_dd)
    
    # Information Ratio (if benchmark provided)
    info_ratio = None
    if benchmark is not None and 'Close' in benchmark:
        benchmark_returns = benchmark['Close'].pct_change()
        active_returns = returns - benchmark_returns
        info_ratio = np.sqrt(settings['periods_per_year']) * active_returns.mean() / active_returns.std()
    
    metrics = {
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar
    }
    
    if info_ratio is not None:
        metrics['information_ratio'] = info_ratio
    
    return metrics

def _calculate_drawdown_metrics(prices: Optional[pd.Series], returns: pd.Series) -> Dict[str, float]:
    """Calculate drawdown metrics."""
    if prices is None:
        prices = (1 + returns).cumprod()
    
    # Rolling maximum
    rolling_max = prices.expanding(min_periods=1).max()
    drawdowns = prices / rolling_max - 1
    
    # Maximum drawdown
    max_drawdown = drawdowns.min()
    
    # Average drawdown
    avg_drawdown = drawdowns[drawdowns < 0].mean()
    
    # Drawdown duration
    is_drawdown = drawdowns < 0
    drawdown_ends = is_drawdown.shift(-1) & ~is_drawdown
    drawdown_starts = ~is_drawdown.shift(1) & is_drawdown
    
    if len(drawdown_starts) > 0 and len(drawdown_ends) > 0:
        durations = []
        for start, end in zip(drawdown_starts.index[drawdown_starts], 
                            drawdown_ends.index[drawdown_ends]):
            durations.append((end - start).days)
        avg_duration = np.mean(durations) if durations else 0
    else:
        avg_duration = 0
    
    return {
        'max_drawdown': max_drawdown,
        'avg_drawdown': avg_drawdown,
        'avg_drawdown_duration': avg_duration
    }

def _calculate_statistical_metrics(returns: pd.Series, settings: Dict[str, Any]) -> Dict[str, float]:
    """Calculate statistical metrics."""
    from scipy import stats
    
    # Basic statistics
    skewness = stats.skew(returns.dropna())
    kurtosis = stats.kurtosis(returns.dropna())
    
    # Autocorrelation
    autocorr = returns.autocorr(lag=1)
    
    # Rolling metrics
    rolling_vol = returns.rolling(window=settings['rolling_window']).std() * np.sqrt(settings['periods_per_year'])
    
    return {
        'skewness': skewness,
        'kurtosis': kurtosis,
        'autocorrelation': autocorr,
        'rolling_volatility': rolling_vol
    }

# Export the functions directly
__all__ = [
    'fetch_market_data',
    'calculate_technical_indicators',
    'generate_plot',
    'calculate_performance_metrics',
    'calculate_metrics'
] 