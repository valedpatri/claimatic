from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    # Secrets
    API_LAYER_KEY: Annotated[str, Field(alias="API_LAYER_KEY")]

    # URLs
    API_LAYER_SENTIMENT_URL: Annotated[str, Field(alias="API_LAYER_SENTIMENT_URL")]

    # DB
    DATABASE_FILE: Annotated[str, Field(alias="DATABASE_FILE")]

    # Language settings
    BASE_LANGUAGE: Annotated[str, Field(alias="BASE_LANGUAGE")]
    CLAIM_LANGUAGE: Annotated[str, Field(alias="CLAIM_LANGUAGE")]

    # Environment
    ENV: Annotated[str, Field(alias="ENV")]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )


settings = Settings()
