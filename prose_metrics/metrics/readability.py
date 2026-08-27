"""Readability and text complexity metrics calculation module."""

import threading
from typing import Final

from spacy.tokens import Doc
from textstat import textstat

from prose_metrics.models.report import ReadabilityMetrics

DEFAULT_READING_SPEED_WPM: Final[int] = 200

SUPPORTED_LANGUAGES: Final[frozenset[str]] = frozenset({"en", "es", "fr", "it", "de", "nl"})

_TEXTSTAT_LOCK = threading.Lock()


def compute_readability_metrics(
    text: str,
    doc: Doc,
    language: str = "en",
    words_per_minute: int = DEFAULT_READING_SPEED_WPM,
) -> ReadabilityMetrics:
    """Compute readability metrics and reading time estimation.

    If language is not English, the Gunning Fog Index will be set to 0.0 since it is not supported by textstat for
    other supported languages.

    Args:
        text (str): The raw text string.
        doc (Doc): The parsed spaCy Doc instance.
        language (str): Language code of the text (ISO 639-1 format). Defaults to "en" for English.
        words_per_minute (int): Average reading speed in words per minute. Defaults to 200.

    Returns:
        ReadabilityMetrics: An instance containing Flesch Reading Ease, Flesch-Kincaid Grade Level, Gunning Fog Index,
            and reading time.

    Raises:
        ValueError: If words_per_minute is less than 1 or if the specified language is not supported.
    """
    lang = language.lower()
    if lang not in SUPPORTED_LANGUAGES:
        msg = f"Readability metrics not supported for language: '{language}'. Supported: {sorted(SUPPORTED_LANGUAGES)}"
        raise ValueError(msg)

    if words_per_minute < 1:
        msg = f"words_per_minute must be >= 1, got {words_per_minute}"
        raise ValueError(msg)

    word_count = sum(1 for tok in doc if not (tok.is_punct or tok.is_space))

    # Guard against empty texts where textstat could divide by zero
    if word_count == 0:
        return ReadabilityMetrics(
            flesch_reading_ease=0.0,
            flesch_kincaid_grade=0.0,
            gunning_fog=0.0,
            estimated_reading_time_minutes=0.0,
        )

    with _TEXTSTAT_LOCK:
        textstat.set_lang(lang)

        try:
            flesch_reading_ease = round(float(textstat.flesch_reading_ease(text)), 2)
            flesch_kincaid_grade = round(float(textstat.flesch_kincaid_grade(text)), 2)
            if lang == "en":
                gunning_fog = round(float(textstat.gunning_fog(text)), 2)
            else:
                gunning_fog = 0.0  # Gunning Fog Index is supported only for English in textstat
        except (ZeroDivisionError, ValueError):
            flesch_reading_ease = 0.0
            flesch_kincaid_grade = 0.0
            gunning_fog = 0.0

    # Reading time in minutes
    estimated_reading_time_minutes = round(word_count / words_per_minute, 2)

    return ReadabilityMetrics(
        flesch_reading_ease=flesch_reading_ease,
        flesch_kincaid_grade=flesch_kincaid_grade,
        gunning_fog=gunning_fog,
        estimated_reading_time_minutes=estimated_reading_time_minutes,
    )
