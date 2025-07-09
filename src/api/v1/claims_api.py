from collections import defaultdict
from http import HTTPStatus
import re

from aiohttp import ClientSession
import aiosqlite
from fastapi import APIRouter
from fastapi import Depends
from fastapi import Form
from fastapi import HTTPException
from loguru import logger

from src.api.dependencies import get_aiohttp_session
from src.api.dependencies import get_db_connection
from src.api.v1.utils.sentiment import analyze_sentiment
from src.api.v1.utils.translate import handle_translation
from src.db.database import add_claim_to_db
from src.db.database import close_claim_by_id
from src.db.database import get_last_hour_claims
from src.llms.categorizer import AsyncClaimCategorizer
from src.schemas.schemas import Claim
from src.schemas.schemas import ClaimRank


claims_router = APIRouter()

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
        sentiment=sentiment,
        category=category,
        warning=warning_message,
    )


@claims_router.get("/open-last-hour")
async def get_open_claims_last_hour(
    db_conn: aiosqlite.Connection = Depends(get_db_connection),
) -> dict[str, list[Claim]]:
    rows = await get_last_hour_claims(db_conn)

    grouped: defaultdict[str, list[Claim]] = defaultdict(list)
    for row in rows:
        claim_obj = Claim.model_validate(row)
        grouped[str(claim_obj.category.value)].append(claim_obj)

    return dict(grouped)


@claims_router.post("/{claim_id}/close")
async def close_claim(
    claim_id: int,
    db_conn: aiosqlite.Connection = Depends(get_db_connection),
) -> None:
    await close_claim_by_id(claim_id, db_conn)
