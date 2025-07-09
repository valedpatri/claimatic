from http import HTTPStatus

from fastapi import APIRouter

from src.schemas.schemas import HealthCheckResponse


health_router = APIRouter()


@health_router.get("/", tags=["Health Check"], status_code=HTTPStatus.OK)
async def read_root() -> HealthCheckResponse:
    """A simple health check endpoint."""
    return HealthCheckResponse(
        message="Clients Claims API is running. API info on: /docs"
    )
