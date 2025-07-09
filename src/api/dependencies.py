from contextlib import asynccontextmanager
from typing import AsyncGenerator
from typing import cast

from aiohttp import ClientSession
import aiosqlite
from fastapi import Request


def get_aiohttp_session(request: Request) -> ClientSession:
    """Dependency to get the aiohttp session from app state."""
    return cast(ClientSession, request.app.state.aiohttp_session)


@asynccontextmanager
async def get_db_connection_context(
    request: Request,
) -> AsyncGenerator[aiosqlite.Connection, None]:
    async with aiosqlite.connect(request.app.state.db_file) as connection:
        connection.row_factory = aiosqlite.Row
        yield connection


async def get_db_connection(
    request: Request,
) -> AsyncGenerator[aiosqlite.Connection, None]:
    async with get_db_connection_context(request) as conn:
        yield conn
