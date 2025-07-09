from contextlib import asynccontextmanager
from typing import AsyncIterator

from aiohttp import ClientSession
from fastapi import FastAPI
from loguru import logger

from src.api.v1.claims_api import claims_router
from src.api.v1.health_check import health_router
from src.core.config import settings
from src.core.logging_config import configure_logging
from src.db.database import init_db


DATABASE_FILE = settings.DATABASE_FILE


def create_app() -> FastAPI:
    """Application factory function"""
    app = FastAPI(lifespan=lifespan, title="Claim Ranking API")
    app.include_router(health_router)
    app.include_router(claims_router, prefix="/claims")

    return app


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manages app startup and shutdown events with proper logging setup"""
    configure_logging()
    logger.info("Application starting up...")

    try:
        logger.info(f"Setting up database with file: {DATABASE_FILE}")
        app.state.db_file = DATABASE_FILE
        await init_db(DATABASE_FILE)

        app.state.aiohttp_session = ClientSession()

        yield

    except Exception as e:
        logger.critical(f"Application startup failed: {e}")
        raise

    finally:
        logger.info("Application shutting down...")

        if hasattr(app.state, "aiohttp_session"):
            await app.state.aiohttp_session.close()
            logger.info("AIOHTTP session closed.")


app = create_app()
