from datetime import datetime
from datetime import timedelta
from datetime import timezone
import sqlite3

import aiosqlite
from fastapi import HTTPException
from loguru import logger


async def init_db(db_path: str) -> None:
    """Initializes the database and creates the 'claims' table if it doesn't exist."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS claims (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                translation TEXT DEFAULT 'Not translated',
                status TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                category TEXT NOT NULL
            )
            """
        )
        await db.commit()
    logger.info("Database initialized successfully.")


async def add_claim_to_db(
    text: str,
    translation: str | None,
    status: str,
    sentiment: str,
    category: str,
    db_conn: aiosqlite.Connection,
) -> int | None:
    """
    Adds a new claim to the database.
    Returns the new claim's ID on success, or None on failure.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    translation = translation or "Not translated"
    try:
        cursor = await db_conn.execute(
            """
                INSERT INTO claims (
                    text, translation, status, timestamp, sentiment, category
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
            (text, translation, status, timestamp, sentiment, category),
        )
        await db_conn.commit()
        new_id = cursor.lastrowid

        if new_id is not None:
            logger.success(f"Claim saved to database with ID: {new_id}")
            return new_id
        else:
            logger.error("Database insert succeeded but did not return a new ID.")
            return None

    except sqlite3.Error as e:
        logger.exception(f"A database error occurred while adding claim: {e}")
        return None


async def get_last_hour_claims(db_conn: aiosqlite.Connection) -> list[aiosqlite.Row]:
    """
    Fetches all 'open' claims of category 'SERVICE', 'PAYMENT', or 'OTHER'
    from the last hour.

    Args:
        db_conn: An active aiosqlite connection from the connection pool.

    Returns:
        A list of claims, where each claim is an aiosqlite.Row object (dict-like).
    """
    logger.info("Fetching claims from the last hour...")

    now = datetime.now(timezone.utc)
    one_hour_ago = now - timedelta(hours=1)

    count_query = """
        SELECT COUNT(*)
        FROM claims
        WHERE status = 'open'
          AND category IN ('SERVICE', 'PAYMENT', 'OTHER')
          AND datetime(timestamp) >= datetime(?)
    """
    async with db_conn.execute(
        count_query, (one_hour_ago.isoformat(),)
    ) as count_cursor:
        result = await count_cursor.fetchone()
        count = result[0] if result else 0
        logger.info(f"Found {count} claims in the last hour.")

    fetch_query = """
        SELECT id, timestamp, text, sentiment, category, status
        FROM claims
        WHERE status = 'open'
          AND category IN ('SERVICE', 'PAYMENT', 'OTHER')
          AND datetime(timestamp) >= datetime(?)
        ORDER BY timestamp DESC
    """
    async with db_conn.execute(
        fetch_query, (one_hour_ago.isoformat(),)
    ) as fetch_cursor:
        rows = await fetch_cursor.fetchall()
        return [row for row in rows]


async def close_claim_by_id(claim_id: int, db_conn: aiosqlite.Connection) -> None:
    """Updates the status of a claim to 'closed' by its ID."""
    cursor = await db_conn.execute(
        "UPDATE claims SET status = 'closed' WHERE id = ?", (claim_id,)
    )
    await db_conn.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Claim not found")

    logger.info(f"Claim with ID {claim_id} marked as closed.")
