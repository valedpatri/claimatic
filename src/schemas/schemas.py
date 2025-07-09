from sqlite3 import Row
from typing import Any
from typing import Literal
from typing import Self

from pydantic import BaseModel
from pydantic import model_validator


SentimentType = Literal["Positive", "Negative", "Neutral", "Unknown"]
ClaimCategory = Literal["SERVICE", "PAYMENT", "ACCOUNT", "OTHER", "AI_UNAVAILABLE"]
SENTIMENT_PRIORITY = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2,
}


class HealthCheckResponse(BaseModel):
    message: str


class OllamaMessage(BaseModel):
    role: str
    content: str


class OllamaResponse(BaseModel):
    model: str
    message: OllamaMessage


class Claim(BaseModel):
    id: int
    timestamp: str
    text: str
    sentiment: str
    category: ClaimCategory | None

    @model_validator(mode="before")
    @classmethod
    def validate_from_row(cls: type[Self], data: Any) -> Any:
        """
        Convert a `sqlite3.Row` to a `dict` before standard validation.
        """
        if isinstance(data, Row):
            return dict(data)
        return data


class ClaimRank(BaseModel):
    """
    Represents a claim with validated status, sentiment, and category.
    """

    id: int
    status: Literal["open", "closed"] = "open"
    sentiment: str
    category: ClaimCategory
    warning: str | None
