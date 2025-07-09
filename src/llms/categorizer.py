import re

import aiohttp
from loguru import logger
from pydantic import ValidationError

from src.schemas.keywords import ACCOUNT_KEYWORDS
from src.schemas.keywords import PAYMENT_KEYWORDS
from src.schemas.keywords import SERVICE_KEYWORDS
from src.schemas.schemas import OllamaResponse


DEFAULT_KEYWORD_MAP: dict[str, set[str]] = {
    "PAYMENT": PAYMENT_KEYWORDS,
    "SERVICE": SERVICE_KEYWORDS,
    "ACCOUNT": ACCOUNT_KEYWORDS,
}
DEFAULT_AI_CATEGORIES = ["PAYMENT", "SERVICE", "OTHER"]


class AsyncClaimCategorizer:
    """
    An async, multi-stage categorizer designed for use with FastAPI.
    It uses aiohttp to communicate with a backend Ollama service.
    """

    def __init__(
        self,
        keyword_map: dict[str, set[str]] = DEFAULT_KEYWORD_MAP,
        ai_categories: list[str] = DEFAULT_AI_CATEGORIES,
        ollama_model: str = "mistral",
        ollama_host: str = "http://localhost:11434",
    ):
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

    async def categorize(self, claim_text: str) -> str:
        """
        Performs the full async categorization process.
        """
        self.logger.info(f"--- New Async Claim: '{claim_text}' ---")

        # --- Stage 1: Synchronous Keyword Detection (it's fast enough) ---
        claim_words = self._preprocess_text(claim_text)
        for category, keywords in self.keyword_map.items():
            if not keywords.isdisjoint(claim_words):
                self.logger.info(
                    f"Stage 1 Result: {category}. Final Category: {category}\n"
                )
                return category

        # --- Stage 2: Asynchronous AI Categorization ---
        async with aiohttp.ClientSession() as session:
            category = await self._categorize_with_ai(claim_text, session)

        self.logger.info(f"Final Category: {category}\n")
        return category
