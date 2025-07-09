from aiohttp import ClientSession
from loguru import logger

from src.core.config import settings
from src.schemas.schemas import Sentiment


API_LAYER_SENTIMENT_URL = settings.API_LAYER_SENTIMENT_URL
API_LAYER_KEY = settings.API_LAYER_KEY


async def analyze_sentiment(
    claim: str, aiohttp_session: ClientSession
) -> tuple[Sentiment, str | None]:
    """
    Analyzes sentiment using an external API and returns a validated Sentiment enum.
    """
    headers = {"apikey": API_LAYER_KEY}
    payload = claim.encode("utf-8")
    sentiment: Sentiment = Sentiment.UNKNOWN
    warning_message: str | None = None

    try:
        async with aiohttp_session.post(
            url=API_LAYER_SENTIMENT_URL,
            headers=headers,
            data=payload,
        ) as resp:
            resp.raise_for_status()
            sentiment_data = await resp.json()
            api_sentiment_str = sentiment_data.get("sentiment", "Neutral").capitalize()
            try:
                sentiment = Sentiment(api_sentiment_str)
                logger.info(f"API sentiment analysis result: {sentiment.value}")
            except ValueError:
                sentiment = Sentiment.UNKNOWN
                warning_message = (
                    f"API returned an unrecognized sentiment: '{api_sentiment_str}'."
                )
                logger.warning(warning_message)

    except Exception as e:
        logger.warning(
            f"Sentiment analysis failed, defaulting to 'Unknown'. Error: {e}"
        )
        warning_message = "Sentiment analysis failed; value defaulted to 'Unknown'."

    return sentiment, warning_message
