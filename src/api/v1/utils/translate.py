from loguru import logger

from src.core.config import settings
from src.llms.translator import LLMTranslator


CLAIM_LANGUAGE = settings.CLAIM_LANGUAGE
BASE_LANGUAGE = settings.BASE_LANGUAGE


async def handle_translation(claim: str) -> str | None:
    """Handle claim translation if needed"""
    translator = LLMTranslator()
    logger.info(f"Translating query from {CLAIM_LANGUAGE} to '{BASE_LANGUAGE}'.")

    try:
        translated_claim = translator.translate(
            query_text=claim,
            source_lang=CLAIM_LANGUAGE,
            target_lang=BASE_LANGUAGE,
        )

        if not translated_claim:
            logger.warning(
                f"Translation from {CLAIM_LANGUAGE} failed. "
                f"Proceeding with original query."
            )
            return None

        if translated_claim == claim:
            logger.info("Translator returned original query.")

        return translated_claim

    except Exception as e:
        logger.error(f"Translation error: {e}")
        return None
