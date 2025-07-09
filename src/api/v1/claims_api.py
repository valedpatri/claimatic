from collections import defaultdict
from http import HTTPStatus
import re
from typing import cast

from aiohttp import ClientSession
import aiosqlite
from fastapi import APIRouter
from fastapi import Depends
from fastapi import Form
from fastapi import HTTPException
from loguru import logger

from src.api.dependencies import get_aiohttp_session
from src.api.dependencies import get_db_connection
from src.core.config import settings
from src.db.database import add_claim_to_db
from src.db.database import close_claim_by_id
from src.db.database import get_last_hour_claims
from src.llms.categorizer import AsyncClaimCategorizer
from src.llms.translator import LLMTranslator
from src.schemas.schemas import Claim
from src.schemas.schemas import ClaimCategory
from src.schemas.schemas import ClaimRank
from src.schemas.schemas import SentimentType


claims_router = APIRouter()

API_LAYER_SENTIMENT_URL = settings.API_LAYER_SENTIMENT_URL
API_LAYER_KEY = settings.API_LAYER_KEY
CLAIM_LANGUAGE = settings.CLAIM_LANGUAGE
BASE_LANGUAGE = settings.BASE_LANGUAGE

categorizer = AsyncClaimCategorizer()


@claims_router.post(
    "/add-claim", response_model=ClaimRank, status_code=HTTPStatus.CREATED
)
async def rank_claim(
    claim: str = Form(...),
    db_conn: aiosqlite.Connection = Depends(get_db_connection),
    aiohttp_session: ClientSession = Depends(get_aiohttp_session),
) -> ClaimRank:
    """
    Analyzes a claim, saves it to the database, and returns the record.
    """
    if not claim:
        logger.warning("Attempted to process an empty claim.")
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Only non-empty claims are allowed",
        )

    original_claim = claim
    translated_claim = None
    logger.info(f"Received new claim for ranking: '{claim[:70]}...'")

    # Handle translation if needed
    if re.search("[\u0400-\u04ff]", claim):
        translated_claim = await handle_translation(claim)
        if translated_claim and translated_claim != claim:
            claim = translated_claim
            logger.info(f"Translated query for analysis: '{claim[:70]}'")

    # Analyze sentiment
    sentiment, warning_message = await analyze_sentiment(
        claim,
        aiohttp_session,
    )

    # Categorize claim
    category = await categorizer.categorize(claim)

    # Save to a database
    new_id = await add_claim_to_db(
        text=original_claim,
        translation=translated_claim,
        status="open",
        sentiment=sentiment,
        category=category,
        db_conn=db_conn,
    )

    if new_id is None:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Failed to save the claim due to a database error.",
        )

    return ClaimRank(
        id=new_id,
        status="open",
        sentiment=cast(SentimentType, sentiment),
        category=cast(ClaimCategory, category),
        warning=warning_message,
    )


async def handle_translation(claim: str) -> str | None:
    """Handle claim translation if needed"""
    translator = LLMTranslator()
    logger.info(f"Translating query from {CLAIM_LANGUAGE} to '{BASE_LANGUAGE}'.")

    try:
        translated_claim = translator.translate(
            query_text=claim,
            source_lang=CLAIM_LANGUAGE,
            target_lang=BASE_LANGUAGE,
        )

        if not translated_claim:
            logger.warning(
                f"Translation from {CLAIM_LANGUAGE} failed. "
                f"Proceeding with original query."
            )
            return None

        if translated_claim == claim:
            logger.info("Translator returned original query.")

        return translated_claim

    except Exception as e:
        logger.error(f"Translation error: {e}")
        return None


async def analyze_sentiment(
    claim: str, session: ClientSession
) -> tuple[str, str | None]:
    """Analyze sentiment using API Layer"""
    headers = {"apikey": API_LAYER_KEY}
    payload = claim.encode("utf-8")
    sentiment = "Unknown"
    warning_message = None

    try:
        async with session.post(
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


@claims_router.get("/open-last-hour")
async def get_open_claims_last_hour(
    db_conn: aiosqlite.Connection = Depends(get_db_connection),
) -> dict[str, list[Claim]]:
    rows = await get_last_hour_claims(db_conn)

    grouped: defaultdict[str, list[Claim]] = defaultdict(list)
    for row in rows:
        claim_obj = Claim.model_validate(row)
        grouped[str(claim_obj.category)].append(claim_obj)

    return dict(grouped)


@claims_router.post("/{claim_id}/close")
async def close_claim(
    claim_id: int,
    db_conn: aiosqlite.Connection = Depends(get_db_connection),
) -> None:
    await close_claim_by_id(claim_id, db_conn)
