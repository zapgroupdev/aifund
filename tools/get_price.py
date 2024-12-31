import os 
from dotenv import load_dotenv
from typing import Optional, Dict, Any
import logging
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from autogen_core.tools import FunctionTool
from typing_extensions import Annotated

load_dotenv()

async def get_stock_data(
    symbol: Annotated[str, "The stock symbol to get data for (e.g., 'AAPL')"],
    data_type: Annotated[str, "Type of data: 'latest' for current price, 'bars' for OHLCV data"] = "latest",
    timeframe: Annotated[Optional[str], "Time frame for bars: '1Min', '5Min', '15Min', '1Hour', '1Day'"] = "1Day",
    limit: Annotated[Optional[int], "Number of bars to return (max 100)"] = 1
) -> Dict[str, Any]:
    """
    Get stock market data from Alpaca Markets API.
    Returns either the latest price or historical OHLCV bars.
    """
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not api_secret:
        raise ValueError("Alpaca API credentials required for stock data")
    
    try:
        # Initialize client
        client = StockHistoricalDataClient(api_key, api_secret)
        
        if data_type.lower() == "latest":
            # Get latest snapshot
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = client.get_stock_latest_quote(request)
            
            if not quote or symbol not in quote:
                raise ValueError(f"No quote data available for {symbol}")
                
            latest = quote[symbol]
            return {
                "symbol": symbol,
                "timestamp": str(latest.timestamp),
                "ask_price": float(latest.ask_price),
                "ask_size": float(latest.ask_size),
                "bid_price": float(latest.bid_price),
                "bid_size": float(latest.bid_size),
                "mid_price": float((latest.ask_price + latest.bid_price) / 2)
            }
            
        elif data_type.lower() == "bars":
            # Map timeframe string to TimeFrame enum
            timeframe_map = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, "Min"),
                "15Min": TimeFrame(15, "Min"),
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day
            }
            
            if timeframe not in timeframe_map:
                raise ValueError(f"Invalid timeframe. Must be one of: {', '.join(timeframe_map.keys())}")
            
            # Get historical bars
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe_map[timeframe],
                limit=limit
            )
            
            bars = client.get_stock_bars(request)
            
            if not bars or symbol not in bars:
                raise ValueError(f"No bar data available for {symbol}")
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "bars": [{
                    "timestamp": str(bar.timestamp),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                    "vwap": float(bar.vwap),
                    "trade_count": bar.trade_count
                } for bar in bars[symbol]]
            }
        else:
            raise ValueError("data_type must be either 'latest' or 'bars'")
            
    except Exception as e:
        logging.error(f"Error fetching stock data: {str(e)}")
        raise

# Create the stock data tool
stock_data_tool = FunctionTool(
    get_stock_data,
    description="Get stock market data (latest quotes or historical bars) from Alpaca Markets"
)