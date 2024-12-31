import os 
from dotenv import load_dotenv
import requests
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
from typing_extensions import Annotated

load_dotenv()

async def get_stock_sentiment(
    ticker: str,
    topics: Annotated[Optional[List[str]], "Optional list of topics to filter news by"] = None,
    time_from: Annotated[Optional[str], "Start time in YYYYMMDDTHHMM format"] = None,
    time_to: Annotated[Optional[str], "End time in YYYYMMDDTHHMM format"] = None,
) -> Dict[str, Any]:
    """
    Get sentiment analysis for a stock using Alpha Vantage News Sentiment API.
    Returns detailed sentiment analysis including news coverage and market sentiment.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("Alpha Vantage API key not found in environment variables")

    params = {
        "function": "NEWS_SENTIMENT",
        "apikey": api_key,
        "tickers": ticker,
        "sort": "LATEST",
        "limit": 50
    }
    
    if topics:
        params["topics"] = ",".join(topics)
    if time_from:
        params["time_from"] = time_from
    if time_to:
        params["time_to"] = time_to
        
    try:
        response = requests.get("https://www.alphavantage.co/query", params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Error fetching sentiment data: {str(e)}")
        raise

# Create the sentiment analysis tool
sentiment_tool = FunctionTool(
    get_stock_sentiment,
    description="Get market sentiment analysis for a given stock symbol using news and social media data"
)

