"""Thread-safe caching and loading of spaCy pipelines, optimized for text analysis tasks."""

import _thread
import threading
from typing import ClassVar

import spacy
from spacy.language import Language

from prose_metrics.nlp.exceptions import ModelNotFoundError


class SpacyPipelineManager:
    """Singleton manager that caches and configures spaCy pipelines.

    Ensures models are loaded once, with unnecessary pipeline components disabled for maximum performance.
    """

    DEFAULT_MODELS: ClassVar[dict[str, str]] = {
        "en": "en_core_web_sm",
        "fr": "fr_core_news_sm",
    }

    DEFAULT_DISABLED_COMPONENTS: ClassVar[tuple[str, ...]] = ("ner",)

    _instance: ClassVar["SpacyPipelineManager | None"] = None
    _lock: ClassVar[_thread.LockType] = threading.Lock()
    _loaded_pipelines: dict[str, Language]

    def __new__(cls) -> "SpacyPipelineManager":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._loaded_pipelines = {}
        return cls._instance

    def get_pipeline(
        self,
        language: str = "en",
        model_name: str | None = None,
        disable: tuple[str, ...] | None = None,
    ) -> Language:
        """Get or load a cached spaCy Language pipeline.

        Args:
            language (str): Target language code (ISO 639-1, e.g., "en" for English). Defaults to "en".
            model_name (str | None): Optional explicit spaCy model name. If None, uses the default model for the
                specified language. Defaults to None.
            disable (tuple[str, ...] | None): Optional tuple of spaCy pipe components to disable. If None, uses the
                default disabled components ("ner",). Defaults to None.

        Returns:
            Language: The loaded and configured spaCy Language instance.

        Raises:
            ModelNotFoundError: If the requested spaCy model is not installed.
            ValueError: If an unsupported language code is provided without a 'model_name'.
        """
        lang = language.lower()
        target_model = model_name or self.DEFAULT_MODELS.get(lang)

        if not target_model:
            msg = (
                f"Unsupported language code: '{language}'. "
                f"Supported defaults are {list(self.DEFAULT_MODELS.keys())}, "
                "or provide an explicit 'model_name'."
            )
            raise ValueError(msg)

        disabled_pipes = list(disable if disable is not None else self.DEFAULT_DISABLED_COMPONENTS)
        cache_key = f"{target_model}:{','.join(sorted(disabled_pipes))}"

        pipeline: Language | None = self._loaded_pipelines.get(cache_key)
        if pipeline is None:
            with self._lock:
                pipeline = self._loaded_pipelines.get(cache_key)
                if pipeline is None:
                    try:
                        pipeline = spacy.load(target_model, disable=disabled_pipes)
                    except OSError as err:
                        raise ModelNotFoundError(model_name=target_model, language=lang) from err

                    self._loaded_pipelines[cache_key] = pipeline

        return pipeline

    def clear_cache(self) -> None:
        """Clear the cached spaCy pipelines."""
        with self._lock:
            self._loaded_pipelines.clear()
