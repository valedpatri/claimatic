import re
from typing import Mapping
from typing import Sequence

import aiohttp
from loguru import logger
from pydantic import ValidationError

from src.schemas.keywords import ACCOUNT_KEYWORDS
from src.schemas.keywords import PAYMENT_KEYWORDS
from src.schemas.keywords import SERVICE_KEYWORDS
from src.schemas.schemas import ClaimCategory
from src.schemas.schemas import OllamaResponse


DEFAULT_KEYWORD_MAP: dict[ClaimCategory, set[str]] = {
    ClaimCategory.PAYMENT: PAYMENT_KEYWORDS,
    ClaimCategory.SERVICE: SERVICE_KEYWORDS,
    ClaimCategory.ACCOUNT: ACCOUNT_KEYWORDS,
}
DEFAULT_AI_CATEGORIES = [
    ClaimCategory.PAYMENT,
    ClaimCategory.SERVICE,
    ClaimCategory.OTHER,
]


class AsyncClaimCategorizer:
    """
    An async, multi-stage categorizer designed for use with FastAPI.
    It uses aiohttp to communicate with a backend Ollama service.
    """

    def __init__(
        self,
        keyword_map: Mapping[ClaimCategory, set[str]] | None = None,
        ai_categories: Sequence[ClaimCategory] | None = None,
        ollama_model: str = "mistral",
        ollama_host: str = "http://localhost:11434",
    ):
        if keyword_map is None:
            keyword_map = DEFAULT_KEYWORD_MAP
        if ai_categories is None:
            ai_categories = DEFAULT_AI_CATEGORIES

        self.logger = logger
        self.keyword_map = keyword_map
        self.ai_categories = ai_categories
        self.ollama_model = ollama_model
        self.ollama_endpoint = f"{ollama_host}/api/chat"
        self.logger.info(
            f"AsyncCategorizer initialized for Ollama at {self.ollama_endpoint}"
        )

    def _preprocess_text(self, text: str) -> set[str]:
        """Synchronous helper to preprocess text."""
        text = re.sub(r"[^\w\s]", "", text.lower())
        return set(text.split())

    async def _categorize_with_ai(
        self, claim_text: str, session: aiohttp.ClientSession
    ) -> str:
        """Asynchronously calls the Ollama service for categorization."""
        system_prompt = f"""You are an expert text classification system.
        Classify the user's message into ONE of the following categories.
        Respond with ONLY the category name and nothing else.
        Categories: {", ".join(self.ai_categories)}"""

        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": claim_text},
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        }

        self.logger.info(
            f"Stage 1 failed. Asking local Ollama model ({self.ollama_model})..."
        )
        try:
            async with session.post(self.ollama_endpoint, json=payload) as response:
                response.raise_for_status()
                response_data = await response.json()
                parsed_response = OllamaResponse.model_validate(response_data)
                if not parsed_response.message:
                    self.logger.warning(
                        "Ollama response was valid but contained no 'message' object."
                    )
                    return "OTHER"

                ai_category = parsed_response.message.content.strip()
                if ai_category in self.ai_categories:
                    return ai_category  # Success!
                else:
                    self.logger.warning(
                        f"Ollama returned an unrecognized category: '{ai_category}'."
                    )
                    return "OTHER"

        except ValidationError as e:
            self.logger.error(
                f"Ollama response failed validation. Shape was wrong: {e}"
            )
            return "AI_ERROR"

        except aiohttp.ClientConnectorError:
            self.logger.error(
                "Connection to Ollama failed. Is the Docker container running?"
            )
            return "AI_UNAVAILABLE"

    async def categorize(self, claim_text: str) -> ClaimCategory:
        """
        Performs the full async categorization process,
        returning a valid ClaimCategory enum.
        """
        self.logger.info(f"--- New Async Claim: '{claim_text}' ---")

        # --- Stage 1: Synchronous Keyword Detection ---
        claim_words = self._preprocess_text(claim_text)
        for category, keywords in self.keyword_map.items():
            if not keywords.isdisjoint(claim_words):
                self.logger.info(
                    f"Stage 1 Result: {category.value}."
                    f" Final Category: {category.value}\n"
                )
                return category

        # --- Stage 2: Asynchronous AI Categorization ---
        self.logger.info("Stage 1 did not find a match. Proceeding to Stage 2 (AI).")

        final_category: ClaimCategory
        try:
            async with aiohttp.ClientSession() as session:
                ai_category_str = await self._categorize_with_ai(claim_text, session)

            try:
                final_category = ClaimCategory(ai_category_str)
                self.logger.info(f"Stage 2 (AI) Result: {final_category.value}")
            except ValueError:
                self.logger.warning(
                    f"AI returned an unrecognized category: '{ai_category_str}'."
                    f" Defaulting to OTHER."
                )
                final_category = ClaimCategory.OTHER

        except Exception as e:
            self.logger.error(
                f"AI categorization failed: {e}. Defaulting to AI_UNAVAILABLE."
            )
            final_category = ClaimCategory.AI_UNAVAILABLE

        self.logger.info(f"Final Category: {final_category.value}\n")
        return final_category
