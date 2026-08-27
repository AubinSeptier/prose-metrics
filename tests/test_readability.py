"""Unit tests for readability metrics."""

import pytest
from spacy.language import Language

from prose_metrics.metrics.readability import compute_readability_metrics
from prose_metrics.nlp.pipeline import SpacyPipelineManager


@pytest.fixture(scope="module")
def nlp() -> Language:
    """Provide a cached spaCy pipeline for tests."""
    return SpacyPipelineManager().get_pipeline(language="en")


@pytest.fixture(scope="module")
def nlp_fr() -> Language:
    """Provide a cached spaCy pipeline for French tests."""
    return SpacyPipelineManager().get_pipeline(language="fr")


def test_empty_text_readability(nlp: Language) -> None:
    """Check degenerate and empty text returns 0.0 without exceptions."""
    text = "   \n\n ... "
    doc = nlp(text)
    metrics = compute_readability_metrics(text, doc)

    assert metrics.flesch_reading_ease == 0.0
    assert metrics.flesch_kincaid_grade == 0.0
    assert metrics.gunning_fog == 0.0
    assert metrics.estimated_reading_time_minutes == 0.0


def test_english_readability(nlp: Language) -> None:
    """Check readability calculation on English narrative text."""
    text = (
        "The quick brown fox jumped smoothly over the lazy sleeping dog. "
        "It was a bright and sunny morning in the middle of spring. "
        "Everything in the forest was quiet and peaceful."
    )
    doc = nlp(text)
    metrics = compute_readability_metrics(text, doc, language="en")

    assert metrics.flesch_reading_ease > 0.0
    assert metrics.flesch_kincaid_grade > 0.0
    assert metrics.gunning_fog > 0.0
    assert metrics.estimated_reading_time_minutes > 0.0


def test_reading_time_calculation(nlp: Language) -> None:
    """Check accurate reading time according to word count and WPM."""
    text = "This is a test sentence." * 100
    doc = nlp(text)
    metrics = compute_readability_metrics(text, doc, words_per_minute=250)

    expected_minutes = round(sum(1 for tok in doc if not (tok.is_punct or tok.is_space)) / 250, 2)
    assert metrics.estimated_reading_time_minutes == expected_minutes


def test_unsupported_language_raises_value_error(nlp: Language) -> None:
    """Check unsupported languages raise ValueError."""
    text = "The quick brown fox jumped over the lazy dog."
    doc = nlp(text)
    with pytest.raises(ValueError, match="Readability metrics not supported for language"):
        compute_readability_metrics(text, doc, language="zh")


def test_invalid_words_per_minute_raises_value_error(nlp: Language) -> None:
    """Check non-positive reading speed raises ValueError."""
    text = "The quick brown fox jumped over the lazy dog."
    doc = nlp(text)
    with pytest.raises(ValueError, match="words_per_minute must be >= 1"):
        compute_readability_metrics(text, doc, words_per_minute=0)


def test_non_english_gunning_fog_is_zero(nlp: Language) -> None:
    """Check Gunning Fog Index is zero for non-English languages while Flesch scores remain."""
    text = "Le petit chat noir dort tranquillement sur le tapis rouge de la maison."
    doc = nlp(text)
    metrics = compute_readability_metrics(text, doc, language="fr")

    assert metrics.gunning_fog == 0.0
    assert metrics.flesch_reading_ease > 0.0
    assert metrics.flesch_kincaid_grade > 0.0


def test_language_code_is_case_insensitive(nlp: Language) -> None:
    """Check language codes are normalized before validation."""
    text = "The quick brown fox jumped over the lazy dog."
    doc = nlp(text)
    metrics_lower = compute_readability_metrics(text, doc, language="en")
    metrics_upper = compute_readability_metrics(text, doc, language="EN")

    assert metrics_lower == metrics_upper


def test_concurrent_multilingual_readability(nlp: Language) -> None:
    """Check concurrent calls with different languages never corrupt textstat's global state."""
    import threading

    text = "The quick brown fox jumped over the lazy dog near the old river mine."
    doc = nlp(text)
    languages = ("en", "fr", "de", "es")
    expected = {lang: compute_readability_metrics(text, doc, language=lang) for lang in languages}
    errors: list[BaseException] = []

    def worker(lang: str) -> None:
        try:
            for _ in range(10):
                assert compute_readability_metrics(text, doc, language=lang) == expected[lang]
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(lang,)) for lang in languages for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors


def test_french_readability(nlp_fr: Language) -> None:
    """Check readability calculation on French narrative text."""
    text = (
        "Le renard brun et vif a sauté avec agilité par-dessus le chien paresseux qui dormait. "
        "C'était une matinée radieuse et ensoleillée, en plein printemps. "
        "Tout était calme et paisible dans la forêt."
    )
    doc = nlp_fr(text)
    metrics = compute_readability_metrics(text, doc, language="fr")

    assert metrics.flesch_reading_ease > 0.0
    assert metrics.flesch_kincaid_grade > 0.0
    assert metrics.gunning_fog == 0.0
    assert metrics.estimated_reading_time_minutes > 0.0
