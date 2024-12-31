import os 
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List, Union
import logging
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, 
    LimitOrderRequest, 
    StopOrderRequest, 
    TrailingStopOrderRequest,
    GetAssetsRequest,
    GetOrdersRequest,
    CreateWatchlistRequest,
    UpdateWatchlistRequest
)
from alpaca.trading.enums import (
    OrderSide, 
    TimeInForce, 
    AssetClass, 
    AssetStatus,
    QueryOrderStatus
)
from autogen_core.tools import FunctionTool
from typing_extensions import Annotated

load_dotenv()

async def execute_trade(
    symbol: Annotated[str, "The stock/crypto symbol to trade"],
    side: Annotated[str, "Buy or sell"] = "buy",
    order_type: Annotated[str, "Type of order: 'market', 'limit', 'stop', or 'trailing_stop'"] = "market",
    quantity: Annotated[Optional[float], "Quantity to trade (use either quantity or notional)"] = None,
    notional: Annotated[Optional[float], "Dollar amount to trade (use either quantity or notional)"] = None,
    limit_price: Annotated[Optional[float], "Limit price for limit orders"] = None,
    stop_price: Annotated[Optional[float], "Stop price for stop orders"] = None,
    trail_percent: Annotated[Optional[float], "Trail percent for trailing stop orders"] = None,
    trail_price: Annotated[Optional[float], "Trail price for trailing stop orders"] = None,
    time_in_force: Annotated[str, "Time in force: 'day', 'gtc', 'ioc', or 'fok'"] = "day"
) -> Dict[str, Any]:
    """
    Execute a trade order using Alpaca Trading API.
    Returns the order details and execution status.
    """
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not api_secret:
        raise ValueError("Alpaca API credentials not found in environment variables")
    
    # Input validation
    side = side.upper()
    if side not in ["BUY", "SELL"]:
        raise ValueError("Side must be either 'buy' or 'sell'")
    
    order_type = order_type.lower()
    if order_type not in ["market", "limit", "stop", "trailing_stop"]:
        raise ValueError("Order type must be one of: 'market', 'limit', 'stop', 'trailing_stop'")
        
    if order_type == "limit" and limit_price is None:
        raise ValueError("Limit price is required for limit orders")
        
    if order_type == "stop" and stop_price is None:
        raise ValueError("Stop price is required for stop orders")
        
    if order_type == "trailing_stop" and trail_percent is None and trail_price is None:
        raise ValueError("Either trail_percent or trail_price is required for trailing stop orders")
        
    if quantity is None and notional is None:
        raise ValueError("Either quantity or notional amount must be specified")
        
    if quantity is not None and notional is not None:
        raise ValueError("Cannot specify both quantity and notional amount")
    
    # Map time in force
    time_force_map = {
        "day": TimeInForce.DAY,
        "gtc": TimeInForce.GTC,
        "ioc": TimeInForce.IOC,
        "fok": TimeInForce.FOK
    }
    time_in_force = time_force_map.get(time_in_force.lower())
    if not time_in_force:
        raise ValueError("Invalid time in force value")
    
    try:
        # Initialize Alpaca client (paper trading for safety)
        trading_client = TradingClient(api_key, api_secret, paper=True)
        
        # Prepare order data
        order_data = None
        if order_type == "market":
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                notional=notional,
                side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                time_in_force=time_in_force
            )
        elif order_type == "limit":
            order_data = LimitOrderRequest(
                symbol=symbol,
                limit_price=limit_price,
                qty=quantity,
                notional=notional,
                side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                time_in_force=time_in_force
            )
        elif order_type == "stop":
            order_data = StopOrderRequest(
                symbol=symbol,
                stop_price=stop_price,
                qty=quantity,
                notional=notional,
                side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                time_in_force=time_in_force
            )
        else:  # trailing_stop
            order_data = TrailingStopOrderRequest(
                symbol=symbol,
                trail_percent=trail_percent,
                trail_price=trail_price,
                qty=quantity,
                notional=notional,
                side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                time_in_force=time_in_force
            )
        
        # Submit the order
        order = trading_client.submit_order(order_data=order_data)
        
        # Format response
        return {
            "order_id": order.id,
            "client_order_id": order.client_order_id,
            "symbol": order.symbol,
            "side": order.side,
            "type": order.type,
            "quantity": float(order.qty) if order.qty else None,
            "notional": float(order.notional) if order.notional else None,
            "limit_price": float(order.limit_price) if hasattr(order, 'limit_price') and order.limit_price else None,
            "stop_price": float(order.stop_price) if hasattr(order, 'stop_price') and order.stop_price else None,
            "trail_price": float(order.trail_price) if hasattr(order, 'trail_price') and order.trail_price else None,
            "trail_percent": float(order.trail_percent) if hasattr(order, 'trail_percent') and order.trail_percent else None,
            "status": order.status,
            "created_at": str(order.created_at),
            "submitted_at": str(order.submitted_at),
            "filled_at": str(order.filled_at) if order.filled_at else None,
            "filled_qty": float(order.filled_qty) if order.filled_qty else None,
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None
        }
        
    except Exception as e:
        logging.error(f"Error executing trade: {str(e)}")
        raise

async def cancel_orders(
    order_id: Annotated[Optional[str], "Specific order ID to cancel. If not provided, cancels all open orders"] = None
) -> Dict[str, Any]:
    """
    Cancel a specific order by order_id or all open orders.
    Returns the cancellation status and details.
    """
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not api_secret:
        raise ValueError("Alpaca API credentials not found in environment variables")
    
    try:
        # Initialize Alpaca client (paper trading for safety)
        trading_client = TradingClient(api_key, api_secret, paper=True)
        
        if order_id:
            # Cancel specific order
            result = trading_client.cancel_order(order_id=order_id)
            return {
                "status": "success",
                "message": f"Order {order_id} cancelled successfully",
                "cancelled_orders": [order_id]
            }
        else:
            # Cancel all open orders
            cancelled = trading_client.cancel_orders()
            return {
                "status": "success",
                "message": "All open orders cancelled successfully",
                "cancelled_orders": [order.id for order in cancelled]
            }
            
    except Exception as e:
        logging.error(f"Error cancelling orders: {str(e)}")
        raise

async def get_account_info() -> Dict[str, Any]:
    """
    Get account information including buying power, portfolio value, and status.
    """
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not api_secret:
        raise ValueError("Alpaca API credentials not found in environment variables")
    
    try:
        trading_client = TradingClient(api_key, api_secret, paper=True)
        account = trading_client.get_account()
        
        return {
            "id": account.id,
            "status": account.status,
            "currency": account.currency,
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "pattern_day_trader": account.pattern_day_trader,
            "trading_blocked": account.trading_blocked,
            "transfers_blocked": account.transfers_blocked,
            "account_blocked": account.account_blocked,
            "created_at": str(account.created_at),
            "multiplier": account.multiplier,
            "equity": float(account.equity),
            "last_equity": float(account.last_equity),
            "long_market_value": float(account.long_market_value),
            "short_market_value": float(account.short_market_value),
            "initial_margin": float(account.initial_margin),
            "maintenance_margin": float(account.maintenance_margin),
            "last_maintenance_margin": float(account.last_maintenance_margin),
            "daytrading_buying_power": float(account.daytrading_buying_power),
            "regt_buying_power": float(account.regt_buying_power)
        }
        
    except Exception as e:
        logging.error(f"Error getting account info: {str(e)}")
        raise

async def get_assets(
    asset_class: Annotated[Optional[str], "Filter by asset class: 'us_equity' or 'crypto'"] = None,
    status: Annotated[Optional[str], "Filter by asset status: 'active' or 'inactive'"] = None
) -> List[Dict[str, Any]]:
    """
    Get list of assets available for trading.
    """
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not api_secret:
        raise ValueError("Alpaca API credentials not found in environment variables")
    
    try:
        trading_client = TradingClient(api_key, api_secret, paper=True)
        
        # Prepare search parameters
        search_params = {}
        if asset_class:
            asset_class = asset_class.upper()
            if asset_class == "US_EQUITY":
                search_params["asset_class"] = AssetClass.US_EQUITY
            elif asset_class == "CRYPTO":
                search_params["asset_class"] = AssetClass.CRYPTO
            else:
                raise ValueError("Asset class must be either 'us_equity' or 'crypto'")
                
        if status:
            status = status.upper()
            if status == "ACTIVE":
                search_params["status"] = AssetStatus.ACTIVE
            elif status == "INACTIVE":
                search_params["status"] = AssetStatus.INACTIVE
            else:
                raise ValueError("Status must be either 'active' or 'inactive'")
        
        # Get assets with filters if any
        request = GetAssetsRequest(**search_params) if search_params else None
        assets = trading_client.get_all_assets(request)
        
        return [{
            "id": asset.id,
            "class": asset.asset_class,
            "exchange": asset.exchange,
            "symbol": asset.symbol,
            "name": asset.name,
            "status": asset.status,
            "tradable": asset.tradable,
            "marginable": asset.marginable,
            "shortable": asset.shortable,
            "easy_to_borrow": asset.easy_to_borrow,
            "fractionable": asset.fractionable
        } for asset in assets]
        
    except Exception as e:
        logging.error(f"Error getting assets: {str(e)}")
        raise

async def get_positions(
    symbol: Annotated[Optional[str], "Symbol to get position for"] = None
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Get all positions or a specific position.
    """
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not api_secret:
        raise ValueError("Alpaca API credentials not found in environment variables")
    
    try:
        trading_client = TradingClient(api_key, api_secret, paper=True)
        
        if symbol:
            # Get specific position
            position = trading_client.get_open_position(symbol_or_asset_id=symbol)
            return {
                "symbol": position.symbol,
                "qty": float(position.qty),
                "market_value": float(position.market_value),
                "avg_entry_price": float(position.avg_entry_price),
                "unrealized_pl": float(position.unrealized_pl),
                "asset_class": str(position.asset_class)
            }
        else:
            # Get all positions
            positions = trading_client.get_all_positions()
            return [{
                "symbol": pos.symbol,
                "qty": float(pos.qty),
                "market_value": float(pos.market_value),
                "avg_entry_price": float(pos.avg_entry_price),
                "unrealized_pl": float(pos.unrealized_pl),
                "asset_class": str(pos.asset_class)
            } for pos in positions]
            
    except Exception as e:
        logging.error(f"Error getting positions: {str(e)}")
        raise

async def close_positions(
    symbol: Annotated[Optional[str], "Symbol to close position for"] = None,
    cancel_orders: Annotated[bool, "Whether to cancel orders"] = False
) -> Dict[str, Any]:
    """
    Close all positions or a specific position.
    """
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not api_secret:
        raise ValueError("Alpaca API credentials not found in environment variables")
    
    try:
        trading_client = TradingClient(api_key, api_secret, paper=True)
        
        if symbol:
            # Close specific position
            response = trading_client.close_position(symbol_or_asset_id=symbol)
            return {
                "status": "success",
                "message": f"Position for {symbol} closed successfully",
                "symbol": symbol
            }
        else:
            # Close all positions
            response = trading_client.close_all_positions(cancel_orders=cancel_orders)
            return {
                "status": "success",
                "message": "All positions closed successfully",
                "cancelled_orders": cancel_orders
            }
            
    except Exception as e:
        logging.error(f"Error closing positions: {str(e)}")
        raise

async def manage_watchlist(
    action: Annotated[str, "Action to perform: 'create', 'get', 'update', 'delete'"],
    name: Annotated[str, "Name of the watchlist"],
    symbols: Annotated[Optional[List[str]], "List of symbols to add to watchlist (for create/update)"] = None
) -> Dict[str, Any]:
    """
    Manage watchlists - create, get, update, or delete.
    """
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not api_secret:
        raise ValueError("Alpaca API credentials not found in environment variables")
    
    try:
        trading_client = TradingClient(api_key, api_secret, paper=True)
        
        action = action.lower()
        if action not in ["create", "get", "update", "delete"]:
            raise ValueError("Action must be one of: 'create', 'get', 'update', 'delete'")
            
        if action in ["create", "update"] and not symbols:
            raise ValueError("Symbols list is required for create/update actions")
        
        if action == "create":
            # Create new watchlist
            watchlist_data = CreateWatchlistRequest(
                name=name,
                symbols=symbols
            )
            watchlist = trading_client.create_watchlist(watchlist_data=watchlist_data)
            return {
                "id": str(watchlist.id),
                "name": watchlist.name,
                "symbols": symbols  # Use the provided symbols since they were just added
            }
            
        elif action == "get":
            # First get all watchlists
            watchlists = trading_client.get_watchlists()
            # Find the one with matching name
            watchlist = next((w for w in watchlists if w.name == name), None)
            if not watchlist:
                raise ValueError(f"Watchlist with name '{name}' not found")
            
            # Get the detailed watchlist info
            detailed = trading_client.get_watchlist_by_id(watchlist_id=watchlist.id)
            return {
                "id": str(detailed.id),
                "name": detailed.name,
                "symbols": [asset.symbol for asset in detailed.assets] if hasattr(detailed, 'assets') else []
            }
            
        elif action == "update":
            # First get watchlist ID by name
            watchlists = trading_client.get_watchlists()
            watchlist = next((w for w in watchlists if w.name == name), None)
            if not watchlist:
                raise ValueError(f"Watchlist with name '{name}' not found")
            
            # Update watchlist
            watchlist_data = UpdateWatchlistRequest(
                name=name,
                symbols=symbols
            )
            updated = trading_client.update_watchlist_by_id(
                watchlist_id=watchlist.id,
                watchlist_data=watchlist_data
            )
            return {
                "id": str(updated.id),
                "name": updated.name,
                "symbols": symbols  # Use the provided symbols since they were just updated
            }
            
        else:  # delete
            # First get watchlist ID by name
            watchlists = trading_client.get_watchlists()
            watchlist = next((w for w in watchlists if w.name == name), None)
            if watchlist:
                trading_client.delete_watchlist_by_id(watchlist_id=watchlist.id)
            return {
                "status": "success",
                "message": f"Watchlist '{name}' deleted successfully"
            }
            
    except Exception as e:
        logging.error(f"Error managing watchlist: {str(e)}")
        raise

async def get_all_watchlists() -> List[Dict[str, Any]]:
    """
    Get all watchlists.
    """
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not api_secret:
        raise ValueError("Alpaca API credentials not found in environment variables")
    
    try:
        trading_client = TradingClient(api_key, api_secret, paper=True)
        watchlists = trading_client.get_watchlists()
        
        result = []
        for w in watchlists:
            # Get detailed watchlist info to access assets
            detailed = trading_client.get_watchlist_by_id(watchlist_id=w.id)
            result.append({
                "id": str(w.id),
                "name": w.name,
                "symbols": [asset.symbol for asset in detailed.assets] if hasattr(detailed, 'assets') else []
            })
        return result
        
    except Exception as e:
        logging.error(f"Error getting watchlists: {str(e)}")
        raise

# Create the order execution tool
order_tool = FunctionTool(
    execute_trade,
    description="Execute trading orders (market or limit) for stocks and crypto using Alpaca Trading"
)

# Create the order cancellation tool
cancel_tool = FunctionTool(
    cancel_orders,
    description="Cancel a specific order by order_id or all open orders"
)

# Create the account info tool
account_tool = FunctionTool(
    get_account_info,
    description="Get detailed account information including buying power, portfolio value, and account status"
)

# Create the assets tool
assets_tool = FunctionTool(
    get_assets,
    description="Get list of available assets for trading with optional filtering by asset class and status"
)

# Create the positions tool
positions_tool = FunctionTool(
    get_positions,
    description="Get all positions or a specific position"
)

# Create the close positions tool
close_positions_tool = FunctionTool(
    close_positions,
    description="Close all positions or a specific position"
)

# Create the watchlist management tool
watchlist_tool = FunctionTool(
    manage_watchlist,
    description="Manage watchlists - create, get, update, or delete watchlists"
)

# Create the get all watchlists tool
all_watchlists_tool = FunctionTool(
    get_all_watchlists,
    description="Get all watchlists in the account"
) 