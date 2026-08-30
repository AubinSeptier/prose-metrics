"""Exceptions related to NLP pipeline operations."""


class NLPError(Exception):
    """Base class for exceptions in the NLP pipeline."""

    pass


class ModelNotFoundError(NLPError):
    """Raised when a required spaCy language model is not installed."""

    def __init__(self, model_name: str, language: str) -> None:
        """Initialize the exception with a message, the model name and language.

        Args:
            model_name (str): The name of the spaCy model that was not found.
            language (str): The language code for which the model is required.

        Examples:
            >>> err = ModelNotFoundError("en_core_web_sm", "en")
            >>> err.model_name
            'en_core_web_sm'
            >>> print(err)
            The spaCy model 'en_core_web_sm' for language 'en' is not installed.
            Install it via your package manager or run:
                uv run python -m spacy download en_core_web_sm
        """
        message = (
            f"The spaCy model '{model_name}' for language '{language}' is not installed.\n"
            "Install it via your package manager or run:\n"
            f"    uv run python -m spacy download {model_name}"
        )
        super().__init__(message)
        self.model_name = model_name
        self.language = language
