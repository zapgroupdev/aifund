import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import autogen_core
from autogen_core import AgentRuntime, AgentId, MessageContext, CancellationToken
from agents.port_handler import PortfolioAgent, TextMessage, Reset

@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = AsyncMock()
    
    # Mock thread messages
    messages = AsyncMock()
    messages.create = AsyncMock()
    messages.list = AsyncMock(return_value=MagicMock(
        data=[MagicMock(
            content=[MagicMock(
                type="text",
                text=MagicMock(value="Test response")
            )]
        )]
    ))
    messages.delete = AsyncMock()
    
    # Mock thread runs
    runs = AsyncMock()
    runs.create = AsyncMock(return_value=MagicMock(id="run_123"))
    runs.retrieve = AsyncMock(return_value=MagicMock(
        status="completed",
        required_action=None
    ))
    runs.submit_tool_outputs = AsyncMock()
    
    # Set up client structure
    client.beta.threads.messages = messages
    client.beta.threads.runs = runs
    
    return client

@pytest.fixture
def agent_runtime():
    """Create a mock agent runtime."""
    runtime = MagicMock(spec=AgentRuntime)
    runtime.register_factory = AsyncMock(return_value=AgentId(type="portfolio_agent", key="test"))
    return runtime

@pytest.fixture
async def portfolio_agent_factory(agent_runtime, mock_openai_client):
    """Create a portfolio agent factory for testing."""
    return lambda: PortfolioAgent(
        description="Test Portfolio Agent",
        client=mock_openai_client,
        assistant_id="test_assistant_id",
        thread_id="test_thread_id"
    )

@pytest.fixture
async def portfolio_agent(agent_runtime, portfolio_agent_factory):
    """Create a portfolio agent instance for testing."""
    agent = await PortfolioAgent.register(
        runtime=agent_runtime,
        type="portfolio_agent",
        factory=portfolio_agent_factory
    )
    agent = await agent()  # Get the actual agent instance
    return agent

@pytest.fixture
def message_context():
    """Create a message context for testing."""
    return MessageContext(
        sender=AgentId(type="user", key="test"),
        topic_id="test_topic",
        is_rpc=True,
        cancellation_token=CancellationToken(),
        message_id="test_message"
    )

@pytest.mark.asyncio
async def test_agent_initialization(portfolio_agent):
    """Test that the agent initializes correctly."""
    assert isinstance(portfolio_agent, PortfolioAgent)
    assert portfolio_agent._assistant_id == "test_assistant_id"
    assert portfolio_agent._thread_id == "test_thread_id"
    assert len(portfolio_agent._tools) > 0  # Check tools are registered

@pytest.mark.asyncio
async def test_handle_message(portfolio_agent, message_context, mock_openai_client):
    """Test handling a message."""
    response = await portfolio_agent.handle_message(
        message=TextMessage(content="Test message", source="user"),
        ctx=message_context
    )
    
    # Verify the message flow
    mock_openai_client.beta.threads.messages.create.assert_called_once()
    mock_openai_client.beta.threads.runs.create.assert_called_once()
    assert response.content == "Test response"

@pytest.mark.asyncio
async def test_handle_message_with_tool_calls(portfolio_agent, message_context, mock_openai_client):
    """Test handling a message that requires tool calls."""
    # Mock a run that requires tool calls
    mock_openai_client.beta.threads.runs.retrieve.side_effect = [
        MagicMock(
            status="requires_action",
            required_action=MagicMock(
                submit_tool_outputs=MagicMock(
                    tool_calls=[MagicMock(
                        id="call_123",
                        function=MagicMock(
                            name="get_account_info",
                            arguments="{}"
                        )
                    )]
                )
            )
        ),
        MagicMock(status="completed")
    ]
    
    response = await portfolio_agent.handle_message(
        message=TextMessage(content="Get account info", source="user"),
        ctx=message_context
    )
    
    # Verify tool calls were handled
    mock_openai_client.beta.threads.runs.submit_tool_outputs.assert_called_once()
    assert response.content == "Test response"

@pytest.mark.asyncio
async def test_handle_message_error(portfolio_agent, message_context, mock_openai_client):
    """Test handling a message when an error occurs."""
    mock_openai_client.beta.threads.runs.retrieve.side_effect = Exception("API Error")
    
    with pytest.raises(Exception) as exc_info:
        await portfolio_agent.handle_message(
            message=TextMessage(content="Test message", source="user"),
            ctx=message_context
        )
    assert str(exc_info.value) == "API Error"

@pytest.mark.asyncio
async def test_handle_reset(portfolio_agent, message_context, mock_openai_client):
    """Test handling a reset message."""
    # Mock list response with messages to delete
    mock_openai_client.beta.threads.messages.list.return_value = MagicMock(
        data=[MagicMock(id="msg_123")],
        has_next_page=lambda: False
    )
    
    await portfolio_agent.handle_reset(Reset(), message_context)
    
    # Verify messages were deleted
    mock_openai_client.beta.threads.messages.delete.assert_called_once_with(
        message_id="msg_123",
        thread_id="test_thread_id"
    ) 