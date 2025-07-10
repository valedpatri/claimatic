import importlib.util  # For checking safetensors availability

from loguru import logger
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import Pipeline  # type: ignore[attr-defined]
from transformers import pipeline  # type: ignore[attr-defined]

from src.core.decorators import log_execution


class LLMTranslator:
    """
    Handles translation of a text from a source language to a target language
    using specified Hugging Face language models.
    """

    # Default models, can be expanded
    DEFAULT_MODEL_MAPPING: dict[str, str] = {
        "ru-en": "Helsinki-NLP/opus-mt-ru-en",
        # "en-es": "Helsinki-NLP/opus-mt-en-es", # Example for another pair
    }
    translation_pipelines: dict[str, Pipeline | None]

    def __init__(
        self, model_mapping: dict[str, str] | None = None, device: str = None
    ) -> None:
        """
        Initializes the LLMTranslator with specified translation models.

        Args:
            model_mapping (dict, optional): A dictionary where keys are "source_lang-target_lang"
                                            (e.g., "ru-en") and values are Hugging Face model
                                            identifiers. Defaults to LLMTranslator.DEFAULT_MODEL_MAPPING.
            device (str, optional): The device to run the models on (e.g., "cpu", "cuda", "mps").
                                    If None, transformers pipeline will attempt auto-detection.
        """  # noqa: E501
        self.model_mapping = (
            model_mapping if model_mapping else self.DEFAULT_MODEL_MAPPING
        )
        self.device_to_use = device
        # Stores loaded pipelines, e.g., {"ru-en": pipeline_object}
        self.translation_pipelines = {}

        self._load_models()

    def _load_models(self) -> None:
        """
        Loads the translation models and tokenizers for each language pair
        defined in self.model_mapping and initializes their pipelines.
        """
        _safetensors_available = importlib.util.find_spec("safetensors") is not None
        if not _safetensors_available:
            logger.warning(
                "LLMTranslator: 'safetensors' library not found. "
                "Model loading might fall back to .bin files,"
                " which could trigger PyTorch version errors. "
                "Consider installing with: pip install safetensors"
            )

        for lang_pair, model_name in self.model_mapping.items():
            try:
                logger.info(
                    f"LLMTranslator: Attempting to load model '{model_name}'"
                    f" for '{lang_pair}' "
                    f"with safetensors preference (available: {_safetensors_available}"
                    f")."
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore [no-untyped-call]

                model_obj = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, use_safetensors=_safetensors_available
                )

                # Prepare the keyword arguments for the pipeline
                pipeline_kwargs = {
                    "model": model_obj,
                    "tokenizer": tokenizer,
                }

                # Conditionally add the device argument if it's valid
                if self.device_to_use:
                    if self.device_to_use.startswith("cuda") or self.device_to_use in [
                        "cpu",
                        "mps",
                    ]:
                        pipeline_kwargs["device"] = self.device_to_use
                    else:
                        try:
                            pipeline_kwargs["device"] = int(self.device_to_use)
                        except ValueError:
                            logger.warning(
                                f"LLMTranslator: Invalid device '{self.device_to_use}'"
                                f" for '{model_name}'. Letting pipeline auto-detect."
                            )

                loaded_pipeline = pipeline("translation", **pipeline_kwargs)

                self.translation_pipelines[lang_pair] = loaded_pipeline
                logger.info(
                    f"LLMTranslator: Model '{model_name}' for '{lang_pair}'"
                    f" loaded successfully "
                    f"on device '{loaded_pipeline.device}' using task"
                    f" 'translation'."
                )
            except Exception as e:
                logger.exception(
                    f"LLMTranslator: Error loading translation model '{model_name}'"
                    f" for '{lang_pair}': {e}.\n"
                    f"Ensure the model is valid. Translation for '{lang_pair}'"
                    f" will be unavailable."
                )
                self.translation_pipelines[lang_pair] = None

    @log_execution()
    def translate(
        self, query_text: str, source_lang: str, target_lang: str
    ) -> str | None:
        """
        Translates the given query text from source_lang to target_lang.

        Args:
            query_text (str): The text to translate.
            source_lang (str): The source language code (e.g., "ru").
            target_lang (str): The target language code (e.g., "en").

        Returns:
            str | None: The translated text, or None if translation fails or the
                        required language pair model is not loaded.
                        Returns original query_text if source and target are the same.
        """
        if not query_text:
            logger.warning("LLMTranslator: Empty query text received for translation.")
            return query_text  # Return empty string as is, or None

        if source_lang == target_lang:
            logger.debug(
                f"LLMTranslator: Source and target languages are the same"
                f" ('{source_lang}'). No translation needed."
            )
            return query_text

        lang_pair_key = f"{source_lang}-{target_lang}"
        selected_pipeline = self.translation_pipelines.get(lang_pair_key)

        if not selected_pipeline:
            return None

        try:
            translated_output = selected_pipeline(query_text)

            # Validate the structure of the output
            if not (isinstance(translated_output, list) and translated_output):
                logger.warning(
                    f"LLMTranslator: Unexpected output format from pipeline:"
                    f" {type(translated_output)}"
                )
                return None

            first_result = translated_output[0]
            if not isinstance(first_result, dict):
                logger.warning(
                    f"LLMTranslator: Unexpected item type in pipeline output:"
                    f" {type(first_result)}"
                )
                return None

            raw_translation = first_result.get("translation_text")

            if isinstance(raw_translation, str):
                logger.debug(f"LLMTranslator: Translated text: '{raw_translation}'")
                return raw_translation
            else:
                logger.warning(
                    "LLMTranslator: 'translation_text' key missing"
                    " or not a string in pipeline output."
                )
                return None

        except Exception as e:
            logger.exception(
                f"LLMTranslator: An error occurred during translation: {e}"
            )
            return None


# Example usage for testing this class in isolation:
if __name__ == "__main__":
    try:
        # Metal Performance Shaders (MPS) backend for GPU training acceleration on Mac:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        translator = LLMTranslator(device=device)

        # translator = LLMTranslator() # Uses default ru-en model, auto device

        if translator.translation_pipelines.get("ru-en"):  # Check if ru-en loaded
            test_query_ru = "Привет, мир!"
            translated_text = translator.translate(
                test_query_ru,
                source_lang="ru",
                target_lang="en",
            )
            print(f"Original (ru): '{test_query_ru}'")
            print(f"Translated (en): '{translated_text}'")

            # Test non-translation case
            test_query_en = "Hello, world!"
            translated_text_en = translator.translate(
                test_query_en,
                source_lang="en",
                target_lang="en",
            )
            print(f"Original (en): '{test_query_en}'")
            print(f"Translated (en): '{translated_text_en}'")

            # Test unsupported pair (will return original text due to current fallback)
            translated_text_unsupported = translator.translate(
                test_query_ru,
                source_lang="ru",
                target_lang="es",
            )
            print(f"Original (ru) for es: '{test_query_ru}'")
            # Should be original
            print(f"Translated (es): '{translated_text_unsupported}'")
        else:
            print("Translator pipeline for 'ru-en' failed to load. Cannot run tests.")

    except Exception as main_e:
        print(f"Error in example usage: {main_e}")
