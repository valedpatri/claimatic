from __future__ import annotations

from datetime import datetime
from enum import Enum
from sqlite3 import Row
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import model_validator


class Sentiment(str, Enum):
    """Enumeration for sentiment types with built-in priority logic."""

    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"
    POSITIVE = "Positive"
    UNKNOWN = "Unknown"

    @property
    def priority(self) -> int | None:
        """Returns the priority for sorting, or None if not applicable."""
        priority_map = {
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2,
        }
        return priority_map.get(self.value)


class ClaimCategory(str, Enum):
    """Enumeration for claim categories."""

    SERVICE = "SERVICE"
    PAYMENT = "PAYMENT"
    ACCOUNT = "ACCOUNT"
    OTHER = "OTHER"
    AI_UNAVAILABLE = "AI_UNAVAILABLE"


class HealthCheckResponse(BaseModel):
    message: str


class OllamaMessage(BaseModel):
    # You could even use a Literal here if roles are fixed
    role: Literal["user", "assistant", "system"] | str
    content: str


class OllamaResponse(BaseModel):
    model: str
    message: OllamaMessage | None = None


class BaseClaim(BaseModel):
    """A base model containing fields common to all claim representations."""

    id: int
    sentiment: Sentiment


class Claim(BaseClaim):
    """Represents a raw claim, typically from the database."""

    timestamp: datetime
    text: str
    category: ClaimCategory

    @model_validator(mode="before")
    @classmethod
    def validate_from_row(cls: type[Claim], data: Any) -> Any:
        """Convert a `sqlite3.Row` to a `dict` before standard validation."""
        if isinstance(data, Row):
            return dict(data)
        return data


class ClaimRank(BaseClaim):
    """Represents a processed claim ready for ranking or action."""

    status: Literal["open", "closed", "pending"] = "open"
    category: ClaimCategory
    warning: str | None = None
