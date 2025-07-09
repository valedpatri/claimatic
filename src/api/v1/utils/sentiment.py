from aiohttp import ClientSession
from loguru import logger

from src.core.config import settings


API_LAYER_SENTIMENT_URL = settings.API_LAYER_SENTIMENT_URL
API_LAYER_KEY = settings.API_LAYER_KEY


async def analyze_sentiment(
    claim: str, aiohttp_session: ClientSession
) -> tuple[str, str | None]:
    """Analyze sentiment using API Layer"""
    headers = {"apikey": API_LAYER_KEY}
    payload = claim.encode("utf-8")
    sentiment = "Unknown"
    warning_message = None

    try:
        async with aiohttp_session.post(
            url=API_LAYER_SENTIMENT_URL, headers=headers, data=payload
        ) as resp:
            resp.raise_for_status()
            sentiment_data = await resp.json()
            sentiment = sentiment_data.get("sentiment", "Neutral").capitalize()
            logger.info(f"API sentiment analysis result: {sentiment}")

    except Exception as e:
        logger.warning(
            f"Sentiment analysis failed, defaulting to 'Unknown'. Error: {e}"
        )
        warning_message = "Sentiment analysis failed; value defaulted to 'Unknown'."

    return sentiment, warning_message
