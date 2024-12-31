import os 
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
import logging
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest, GetOrdersRequest
from alpaca.trading.enums import AssetClass, QueryOrderStatus
from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
from typing_extensions import Annotated

load_dotenv()

async def get_portfolio_info(
    info_type: Annotated[str, "Type of information to retrieve: 'account', 'positions', 'orders', or 'all'"] = "all",
    asset_class: Annotated[Optional[str], "Filter by asset class: 'crypto' or 'us_equity'"] = None,
    order_status: Annotated[Optional[str], "Filter orders by status: 'open', 'closed', 'all'"] = None
) -> Dict[str, Any]:
    """
    Get portfolio information from Alpaca Trading API.
    Returns detailed portfolio information including account details, positions, and orders.
    """
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not api_secret:
        raise ValueError("Alpaca API credentials not found in environment variables")
    
    try:
        # Initialize Alpaca client (paper trading for safety)
        trading_client = TradingClient(api_key, api_secret, paper=True)
        
        result = {}
        
        # Get account information
        if info_type in ['account', 'all']:
            account = trading_client.get_account()
            result['account'] = {
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'currency': account.currency,
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked,
                'created_at': str(account.created_at),
                'status': account.status
            }
        
        # Get positions
        if info_type in ['positions', 'all']:
            positions = trading_client.get_all_positions()
            result['positions'] = [{
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'market_value': float(pos.market_value),
                'avg_entry_price': float(pos.avg_entry_price),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc),
                'asset_class': pos.asset_class,
                'side': pos.side
            } for pos in positions]
        
        # Get orders
        if info_type in ['orders', 'all']:
            # Set up order status filter
            status_filter = None
            if order_status:
                status_map = {
                    'open': QueryOrderStatus.OPEN,
                    'closed': QueryOrderStatus.CLOSED,
                    'all': QueryOrderStatus.ALL
                }
                status_filter = status_map.get(order_status.lower())
            
            # Create request parameters
            request_params = GetOrdersRequest(
                status=status_filter
            ) if status_filter else None
            
            # Get orders
            orders = trading_client.get_orders(filter=request_params)
            result['orders'] = [{
                'symbol': order.symbol,
                'qty': float(order.qty) if order.qty else None,
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'submitted_at': str(order.submitted_at),
                'filled_at': str(order.filled_at) if order.filled_at else None,
                'filled_qty': float(order.filled_qty) if order.filled_qty else None,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None
            } for order in orders]
        
        return result
        
    except Exception as e:
        logging.error(f"Error fetching portfolio data: {str(e)}")
        raise

# Create the portfolio tool
portfolio_tool = FunctionTool(
    get_portfolio_info,
    description="Get detailed portfolio information from Alpaca Trading including account details, positions, and orders"
)

