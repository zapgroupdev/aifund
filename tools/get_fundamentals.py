import os
from dotenv import load_dotenv
import requests
from typing import Dict, Any, Optional
from datetime import datetime
from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool

load_dotenv()

async def get_income_statement(
    ticker: str,
) -> Dict[str, Any]:
    """
    Get detailed income statement data from Alpha Vantage.
    Returns quarterly and annual income statements including total revenue, gross profit, operating expenses, etc.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("Alpha Vantage API key not found in environment variables")

    params = {
        "function": "INCOME_STATEMENT",
        "symbol": ticker,
        "apikey": api_key
    }
    
    try:
        response = requests.get("https://www.alphavantage.co/query", params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise ValueError(f"Error fetching income statement data: {str(e)}")

async def get_balance_sheet(
    ticker: str,
) -> Dict[str, Any]:
    """
    Get detailed balance sheet data from Alpha Vantage.
    Returns quarterly and annual balance sheets including assets, liabilities, and shareholders' equity.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("Alpha Vantage API key not found in environment variables")

    params = {
        "function": "BALANCE_SHEET",
        "symbol": ticker,
        "apikey": api_key
    }
    
    try:
        response = requests.get("https://www.alphavantage.co/query", params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise ValueError(f"Error fetching balance sheet data: {str(e)}")

async def get_cash_flow(
    ticker: str,
) -> Dict[str, Any]:
    """
    Get detailed cash flow statement data from Alpha Vantage.
    Returns quarterly and annual cash flow statements including operating, investing, and financing activities.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("Alpha Vantage API key not found in environment variables")

    params = {
        "function": "CASH_FLOW",
        "symbol": ticker,
        "apikey": api_key
    }
    
    try:
        response = requests.get("https://www.alphavantage.co/query", params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise ValueError(f"Error fetching cash flow data: {str(e)}")

async def get_company_overview(
    ticker: str,
) -> Dict[str, Any]:
    """
    Get company overview data from Alpha Vantage.
    Returns general company information including market cap, PE ratio, dividend yield, etc.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("Alpha Vantage API key not found in environment variables")

    params = {
        "function": "OVERVIEW",
        "symbol": ticker,
        "apikey": api_key
    }
    
    try:
        response = requests.get("https://www.alphavantage.co/query", params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise ValueError(f"Error fetching company overview data: {str(e)}")

# Create the fundamental analysis tools
income_statement_tool = FunctionTool(
    get_income_statement,
    description="Get detailed income statement data for a company including revenue, expenses, and profits"
)

balance_sheet_tool = FunctionTool(
    get_balance_sheet,
    description="Get detailed balance sheet data for a company including assets, liabilities, and equity"
)

cash_flow_tool = FunctionTool(
    get_cash_flow,
    description="Get detailed cash flow statement data for a company including operating, investing, and financing activities"
)

company_overview_tool = FunctionTool(
    get_company_overview,
    description="Get company overview data including market cap, PE ratio, dividend yield, and other key metrics"
) 