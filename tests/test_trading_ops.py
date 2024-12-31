import os
import pytest
import asyncio
from dotenv import load_dotenv
from tools.trading_ops import (
    execute_trade,
    cancel_orders,
    get_account_info,
    get_assets,
    get_positions,
    close_positions,
    manage_watchlist,
    get_all_watchlists
)

# Load environment variables
load_dotenv()

# Verify environment variables are set
def test_environment_setup():
    assert os.getenv("ALPACA_API_KEY") is not None, "ALPACA_API_KEY not set"
    assert os.getenv("ALPACA_SECRET_KEY") is not None, "ALPACA_SECRET_KEY not set"

# Test account information
@pytest.mark.asyncio
async def test_get_account_info():
    result = await get_account_info()
    assert isinstance(result, dict)
    assert "buying_power" in result
    assert "portfolio_value" in result
    assert "status" in result
    assert float(result["buying_power"]) >= 0

# Test asset operations
@pytest.mark.asyncio
async def test_get_assets():
    # Test getting all assets
    all_assets = await get_assets()
    assert isinstance(all_assets, list)
    assert len(all_assets) > 0
    
    # Test filtering by asset class
    equity_assets = await get_assets(asset_class="us_equity")
    assert all(asset["class"] == "us_equity" for asset in equity_assets)
    
    # Test filtering by status
    active_assets = await get_assets(status="active")
    assert all(asset["status"] == "active" for asset in active_assets)

# Test order execution
@pytest.mark.asyncio
async def test_order_execution():
    # Cancel any existing orders first
    await cancel_orders()
    await asyncio.sleep(1)  # Wait for cancellation to process
    
    # Test market order
    market_order = await execute_trade(
        symbol="SPY",
        side="buy",
        order_type="market",
        notional=100  # Buy $100 worth
    )
    assert isinstance(market_order, dict)
    assert market_order["symbol"] == "SPY"
    assert market_order["side"] == "buy"
    await asyncio.sleep(1)  # Wait between orders
    
    # Test limit order (set limit price above current price for buy)
    limit_order = await execute_trade(
        symbol="SPY",
        side="buy",
        order_type="limit",
        quantity=1,
        limit_price=1000.0  # Set high to avoid execution
    )
    assert isinstance(limit_order, dict)
    assert limit_order["type"] == "limit"
    await asyncio.sleep(1)  # Wait between orders
    
    # Cancel all orders before proceeding
    await cancel_orders()
    await asyncio.sleep(1)  # Wait for cancellation to process
    
    # Test stop order (set stop price below current price for sell)
    stop_order = await execute_trade(
        symbol="SPY",
        side="sell",
        order_type="stop",
        quantity=1,
        stop_price=100.0  # Set low to avoid execution
    )
    assert isinstance(stop_order, dict)
    assert stop_order["type"] == "stop"
    await asyncio.sleep(1)  # Wait between orders
    
    # Cancel all orders before proceeding
    await cancel_orders()
    await asyncio.sleep(1)  # Wait for cancellation to process
    
    # Test trailing stop order
    trailing_stop = await execute_trade(
        symbol="SPY",
        side="sell",
        order_type="trailing_stop",
        quantity=1,
        trail_percent=5.0
    )
    assert isinstance(trailing_stop, dict)
    assert trailing_stop["type"] == "trailing_stop"
    await asyncio.sleep(1)  # Wait between orders
    
    # Clean up - cancel all orders
    await cancel_orders()

# Test position management
@pytest.mark.asyncio
async def test_position_management():
    # Get all positions
    positions = await get_positions()
    assert isinstance(positions, list)
    
    # Create a test position if none exists
    if len(positions) == 0:
        await execute_trade(
            symbol="SPY",
            side="buy",
            order_type="market",
            notional=100
        )
        await asyncio.sleep(2)  # Wait for order to process
        positions = await get_positions()
    
    if len(positions) > 0:
        # Test getting specific position
        symbol = positions[0]["symbol"]
        position = await get_positions(symbol=symbol)
        assert isinstance(position, dict)
        assert position["symbol"] == symbol
        
        # Test closing specific position
        close_result = await close_positions(symbol=symbol)
        assert isinstance(close_result, dict)
        assert close_result["status"] == "success"
    
    # Test closing all positions
    close_all = await close_positions(cancel_orders=True)
    assert isinstance(close_all, dict)
    assert close_all["status"] == "success"

# Test watchlist management
@pytest.mark.asyncio
async def test_watchlist_management():
    test_watchlist_name = "Test_Watchlist"
    test_symbols = ["SPY", "QQQ", "IWM"]
    
    try:
        # Delete test watchlist if it exists
        await manage_watchlist(action="delete", name=test_watchlist_name)
        await asyncio.sleep(2)  # Wait for deletion to complete
    except:
        pass
    
    # Create watchlist
    create_result = await manage_watchlist(
        action="create",
        name=test_watchlist_name,
        symbols=test_symbols
    )
    assert isinstance(create_result, dict)
    assert create_result["name"] == test_watchlist_name
    assert all(symbol in create_result["symbols"] for symbol in test_symbols)
    await asyncio.sleep(2)  # Wait between operations
    
    # Get watchlist
    get_result = await manage_watchlist(
        action="get",
        name=test_watchlist_name
    )
    assert isinstance(get_result, dict)
    assert get_result["name"] == test_watchlist_name
    assert all(symbol in get_result["symbols"] for symbol in test_symbols)
    await asyncio.sleep(2)  # Wait between operations
    
    # Update watchlist
    updated_symbols = ["SPY", "DIA"]
    update_result = await manage_watchlist(
        action="update",
        name=test_watchlist_name,
        symbols=updated_symbols
    )
    assert isinstance(update_result, dict)
    assert update_result["name"] == test_watchlist_name
    assert all(symbol in update_result["symbols"] for symbol in updated_symbols)
    await asyncio.sleep(2)  # Wait between operations
    
    # Get all watchlists
    all_watchlists = await get_all_watchlists()
    assert isinstance(all_watchlists, list)
    assert any(w["name"] == test_watchlist_name for w in all_watchlists)
    await asyncio.sleep(2)  # Wait between operations
    
    # Delete watchlist
    delete_result = await manage_watchlist(
        action="delete",
        name=test_watchlist_name
    )
    assert isinstance(delete_result, dict)
    assert delete_result["status"] == "success"

# Test error handling
@pytest.mark.asyncio
async def test_error_handling():
    # Test invalid symbol
    with pytest.raises(Exception):
        await execute_trade(
            symbol="INVALID_SYMBOL",
            side="buy",
            order_type="market",
            quantity=1
        )
    
    # Test invalid order type
    with pytest.raises(ValueError):
        await execute_trade(
            symbol="SPY",
            side="buy",
            order_type="invalid_type",
            quantity=1
        )
    
    # Test missing required parameters
    with pytest.raises(ValueError):
        await execute_trade(
            symbol="SPY",
            side="buy",
            order_type="limit"  # Missing limit_price
        )
    
    # Test invalid watchlist action
    with pytest.raises(ValueError):
        await manage_watchlist(
            action="invalid_action",
            name="Test"
        )

if __name__ == "__main__":
    pytest.main(["-v", __file__])