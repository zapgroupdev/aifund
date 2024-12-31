import pytest
from unittest.mock import patch, Mock
from tools.get_sent_av import AlphaVantageSentiment
import os

@pytest.fixture
def sentiment_analyzer():
    # Ensure we have a mock API key for testing
    os.environ["ALPHAVANTAGE_API_KEY"] = "test_key"
    return AlphaVantageSentiment()

def test_init_missing_api_key():
    # Test initialization without API key
    with pytest.raises(ValueError):
        with patch.dict(os.environ, {}, clear=True):
            AlphaVantageSentiment()

def test_validate_datetime_format(sentiment_analyzer):
    # Test datetime format validation
    assert sentiment_analyzer.validate_datetime_format("20240101T1200") == True
    assert sentiment_analyzer.validate_datetime_format("2024-01-01") == False
    assert sentiment_analyzer.validate_datetime_format("invalid") == False

def test_get_sentiment_invalid_topics(sentiment_analyzer):
    # Test with invalid topics
    with pytest.raises(ValueError) as exc_info:
        sentiment_analyzer.get_sentiment(topics=["invalid_topic"])
    assert "Invalid topics" in str(exc_info.value)

def test_get_sentiment_invalid_sort(sentiment_analyzer):
    # Test with invalid sort parameter
    with pytest.raises(ValueError) as exc_info:
        sentiment_analyzer.get_sentiment(sort="INVALID")
    assert "Invalid sort value" in str(exc_info.value)

def test_get_sentiment_invalid_limit(sentiment_analyzer):
    # Test with invalid limit values
    with pytest.raises(ValueError):
        sentiment_analyzer.get_sentiment(limit=0)
    with pytest.raises(ValueError):
        sentiment_analyzer.get_sentiment(limit=1001)

@patch('requests.get')
def test_get_sentiment_successful_call(mock_get, sentiment_analyzer):
    # Mock successful API response
    mock_response = Mock()
    mock_response.json.return_value = {
        "items": [
            {
                "title": "Test News",
                "sentiment_score": 0.8
            }
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    response = sentiment_analyzer.get_sentiment(
        tickers=["AAPL"],
        topics=["technology"],
        time_from="20240101T0000"
    )

    # Verify the response
    assert "items" in response
    assert len(response["items"]) == 1
    assert response["items"][0]["title"] == "Test News"

    # Verify the API was called with correct parameters
    args, kwargs = mock_get.call_args
    assert kwargs["params"]["function"] == "NEWS_SENTIMENT"
    assert kwargs["params"]["tickers"] == "AAPL"
    assert kwargs["params"]["topics"] == "technology"
    assert kwargs["params"]["time_from"] == "20240101T0000"

@patch('requests.get')
def test_get_sentiment_api_error(mock_get, sentiment_analyzer):
    # Mock API error
    mock_get.side_effect = Exception("API Error")

    with pytest.raises(Exception):
        sentiment_analyzer.get_sentiment() 