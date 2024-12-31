import os
import asyncio
from datetime import datetime
from typing import Optional, List, Any
from dataclasses import dataclass
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler
from openai import AsyncClient, AsyncAssistantEventHandler
from pydantic import BaseModel
from tools.trading_ops import (
    order_tool,
    cancel_tool,
    account_tool,
    assets_tool,
    positions_tool,
    close_positions_tool,
    watchlist_tool,
    all_watchlists_tool
)
from tools.get_price import stock_data_tool

# Message Protocol
@dataclass
class TextMessage:
    content: str
    source: str

@dataclass
class Reset:
    pass

# Response Models
class PortfolioSummary(BaseModel):
    total_value: float
    cash_balance: float
    buying_power: float
    equity: float
    margin_used: float
    day_pl_amount: float
    day_pl_percent: float
    account_status: str
    trading_blocked: bool

class Position(BaseModel):
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    pl_amount: float
    pl_percent: float

class WatchlistEntry(BaseModel):
    symbol: str
    last_price: float
    day_change: float
    day_change_percent: float

class Watchlist(BaseModel):
    name: str
    entries: List[WatchlistEntry]

class MarketData(BaseModel):
    symbol: str
    ask_price: float
    ask_size: float
    bid_price: float
    bid_size: float

class OrderConfirmation(BaseModel):
    type: str
    symbol: str
    side: str
    quantity: int
    price: float
    status: str

class OpenOrder(BaseModel):
    symbol: str
    side: str
    type: str
    price: float
    quantity: int
    status: str
    submitted_at: str

class OpenOrders(BaseModel):
    orders: List[OpenOrder]
    total_count: int

class PortfolioAgent(RoutedAgent):
    """An agent implementation that uses the OpenAI Assistant API to generate responses."""

    def __init__(
        self,
        description: str,
        client: AsyncClient,
        assistant_id: str,
        thread_id: str,
        assistant_event_handler_factory: Optional[AsyncAssistantEventHandler] = None,
    ) -> None:
        """Initialize the portfolio agent with OpenAI Assistant capabilities.
        
        Args:
            description (str): The description of the agent.
            client (AsyncClient): The OpenAI async client.
            assistant_id (str): The assistant ID to use.
            thread_id (str): The thread ID to use.
            assistant_event_handler_factory (Optional[AsyncAssistantEventHandler]): Factory for event handler.
        """
        super().__init__(description)
        self._client = client
        self._assistant_id = assistant_id
        self._thread_id = thread_id
        self._assistant_event_handler_factory = assistant_event_handler_factory
        
        # Register tools
        self._tools = [
            order_tool,
            cancel_tool,
            account_tool,
            assets_tool,
            positions_tool,
            close_positions_tool,
            watchlist_tool,
            all_watchlists_tool,
            stock_data_tool
        ]
        
        # Set up system message
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.system_message = f"""You are a professional trading desk manager. Today is {current_date}.

CAPABILITIES:
- Execute various order types:
  * Market orders
  * Limit orders
  * Stop orders
  * Trailing stop orders
- Manage orders:
  * Cancel specific orders
  * Cancel all open orders
- Monitor account:
  * View account status and details
  * Check buying power and margins
  * Track portfolio value and performance
- Manage positions:
  * View all positions or specific positions
  * Close positions (individual or all)
  * Monitor position metrics (P&L, market value)
- Asset operations:
  * Get available assets for trading
  * Filter assets by class (crypto/equity)
  * Check asset tradability and status
- Watchlist management:
  * Create and delete watchlists
  * Update watchlist contents
  * View all watchlists
- Get market data:
  * Real-time stock prices
  * Historical price data
  * OHLCV bars

RESPONSE FORMATTING:
Your responses must be structured according to the request type:

1. For portfolio status, return a PortfolioSummary:
   {{
     "total_value": float,
     "cash_balance": float,
     "buying_power": float,
     "equity": float,
     "margin_used": float,
     "day_pl_amount": float,
     "day_pl_percent": float,
     "account_status": str,
     "trading_blocked": bool
   }}

2. For positions, return a list of Position:
   {{
     "symbol": str,
     "quantity": int,
     "avg_price": float,
     "current_price": float,
     "pl_amount": float,
     "pl_percent": float
   }}

3. For watchlists, return a Watchlist:
   {{
     "name": str,
     "entries": [
       {{
         "symbol": str,
         "last_price": float,
         "day_change": float,
         "day_change_percent": float
       }}
     ]
   }}

4. For market data, return a MarketData:
   {{
     "symbol": str,
     "ask_price": float,
     "ask_size": float,
     "bid_price": float,
     "bid_size": float
   }}

5. For order confirmations, return an OrderConfirmation:
   {{
     "type": str,
     "symbol": str,
     "side": str,
     "quantity": int,
     "price": float,
     "status": str
   }}

6. For open orders, return OpenOrders:
   {{
     "orders": [
       {{
         "symbol": str,
         "side": str,
         "type": str,
         "price": float,
         "quantity": int,
         "status": str,
         "submitted_at": str
       }}
     ],
     "total_count": int
   }}

ORDER EXECUTION WORKFLOW:
1. When placing an order:
   a. First get current price using stock_data_tool and return MarketData
   b. Then execute the order using order_tool and return OrderConfirmation

2. For market orders:
   - Use the current ask price for buys
   - Use the current bid price for sells

3. For limit orders:
   - Buy limits should be below current ask
   - Sell limits should be above current bid

IMPORTANT NOTES:
1. For "show orders" or "get orders" requests, use the order tools to get open orders, NOT the cancel tool
2. Always return structured data according to the models above
3. Never return raw API responses
4. Format dates in a human-readable way (e.g., "Mar 21, 10:30 AM" instead of ISO format)"""

    @message_handler
    async def handle_message(self, message: TextMessage, ctx: MessageContext) -> TextMessage:
        """Handle a message. This method adds the message to the thread and publishes a response."""
        # Save the message to the thread
        await ctx.cancellation_token.link_future(
            asyncio.ensure_future(
                self._client.beta.threads.messages.create(
                    thread_id=self._thread_id,
                    content=f"{self.system_message}\n\nUser: {message.content}",
                    role="user",
                    metadata={"sender": message.source},
                )
            )
        )

        # Generate a response
        if self._assistant_event_handler_factory:
            async with self._client.beta.threads.runs.stream(
                thread_id=self._thread_id,
                assistant_id=self._assistant_id,
                event_handler=self._assistant_event_handler_factory(),
            ) as stream:
                await ctx.cancellation_token.link_future(asyncio.ensure_future(stream.until_done()))
        else:
            # Create and wait for the run to complete
            run = await ctx.cancellation_token.link_future(
                asyncio.ensure_future(
                    self._client.beta.threads.runs.create(
                        thread_id=self._thread_id,
                        assistant_id=self._assistant_id,
                        tools=self._tools
                    )
                )
            )
            while True:
                run = await ctx.cancellation_token.link_future(
                    asyncio.ensure_future(
                        self._client.beta.threads.runs.retrieve(
                            thread_id=self._thread_id,
                            run_id=run.id
                        )
                    )
                )
                if run.status == "completed":
                    break
                elif run.status == "requires_action" and run.required_action:
                    # Handle tool calls
                    tool_outputs = []
                    for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                        tool = next((t for t in self._tools if t.name == tool_call.function.name), None)
                        if tool:
                            result = await tool.function(**tool_call.function.arguments)
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": str(result)
                            })
                    
                    # Submit tool outputs
                    await ctx.cancellation_token.link_future(
                        asyncio.ensure_future(
                            self._client.beta.threads.runs.submit_tool_outputs(
                                thread_id=self._thread_id,
                                run_id=run.id,
                                tool_outputs=tool_outputs
                            )
                        )
                    )
                await asyncio.sleep(1)

        # Get the last message
        messages = await ctx.cancellation_token.link_future(
            asyncio.ensure_future(
                self._client.beta.threads.messages.list(
                    self._thread_id,
                    order="desc",
                    limit=1
                )
            )
        )
        last_message_content = messages.data[0].content

        # Get the text content from the last message
        text_content = [content for content in last_message_content if content.type == "text"]
        if not text_content:
            raise ValueError(f"Expected text content in the last message: {last_message_content}")

        return TextMessage(content=text_content[0].text.value, source=self.metadata["type"])

    @message_handler
    async def handle_reset(self, message: Reset, ctx: MessageContext) -> None:
        """Handle a reset message. This method deletes all messages in the thread."""
        # Get all messages in this thread
        all_msgs: List[str] = []
        while True:
            if not all_msgs:
                msgs = await ctx.cancellation_token.link_future(
                    asyncio.ensure_future(
                        self._client.beta.threads.messages.list(self._thread_id)
                    )
                )
            else:
                msgs = await ctx.cancellation_token.link_future(
                    asyncio.ensure_future(
                        self._client.beta.threads.messages.list(
                            self._thread_id,
                            after=all_msgs[-1]
                        )
                    )
                )
            for msg in msgs.data:
                all_msgs.append(msg.id)
            if not msgs.has_next_page():
                break

        # Delete all the messages
        for msg_id in all_msgs:
            status = await ctx.cancellation_token.link_future(
                asyncio.ensure_future(
                    self._client.beta.threads.messages.delete(
                        message_id=msg_id,
                        thread_id=self._thread_id
                    )
                )
            )
            assert status.deleted is True
