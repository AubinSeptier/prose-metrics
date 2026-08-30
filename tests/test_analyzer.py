"""Integration and performance tests for TextAnalyzer."""

import pytest
from spacy.language import Language

from prose_metrics import TextAnalyzer, analyze
from prose_metrics.models import TextReport
from prose_metrics.nlp.pipeline import SpacyPipelineManager


@pytest.fixture(scope="module")
def nlp() -> Language:
    """Provide a cached spaCy pipeline for tests."""
    return SpacyPipelineManager().get_pipeline(language="en")


def test_analyze_english_test() -> None:
    """Check complete analysis English pipeline."""
    text = (
        "Fog enveloped the cliff. “We have to leave,” he whispered fearfully. "
        "The waves crashed violently against the black rocks below. "
        "Night was falling inexorably over the abandoned kingdom."
    )
    report = analyze(text, language="en")

    assert isinstance(report, TextReport)
    assert report.language == "en"
    assert report.execution_time_seconds > 0.0

    assert report.volume is not None
    assert report.rhythm is not None
    assert report.style is not None
    assert report.vocabulary is not None
    assert report.readability is not None
    assert report.repetition is not None
    assert report.dialogue is not None

    data = report.to_dict()
    assert isinstance(data, dict)
    assert data["volume"]["word_count"] == report.volume.word_count


def test_selective_metrics_calculation() -> None:
    """Check calculating only a subset of metrics."""
    text = "A very simple sentence to test the selection."
    analyzer = TextAnalyzer(language="en")
    report = analyzer.analyze(text, metrics=["volume", "vocabulary"])

    assert report.volume is not None
    assert report.vocabulary is not None
    assert report.rhythm is None
    assert report.style is None
    assert report.readability is None
    assert report.repetition is None
    assert report.dialogue is None

    data = report.to_dict()
    assert "volume" in data
    assert "vocabulary" in data
    assert "rhythm" in data and data["rhythm"] is None


def test_invalid_metric_name_raises_value_error() -> None:
    """Check passing unknown metric name raises ValueError."""
    text = "This is a test sentence."
    analyzer = TextAnalyzer(language="en")

    with pytest.raises(ValueError, match=r"Invalid metric\(s\): \['unknown_metric'\]"):
        analyzer.analyze(text, metrics=["volume", "unknown_metric"])  # type: ignore


def test_performance_3000_words(nlp: Language) -> None:
    """Check performance test ensuring 3,000 words are processed in under 1 second."""
    # Warm up the spaCy pipeline
    _ = nlp

    text = (
        "The knight gazed at the distant horizon with profound sadness. "
        "“There's no hope left,” he whispered softly to his faithful companion. \n\n"
        "The wind was blowing fiercely across the plain."
    )
    text *= 100  # Repeat to create ~3,000 words

    analyzer = TextAnalyzer(language="en")
    report = analyzer.analyze(text, metrics="all")

    assert report.volume is not None
    assert report.volume.word_count >= 3000, f"Word count too low: {report.volume.word_count}"
    assert report.execution_time_seconds < 1.0, f"Execution time exceeded: {report.execution_time_seconds}s"


def test_analyze_french_test() -> None:
    """Check complete analysis French pipeline."""
    text = (
        "Le brouillard enveloppait la falaise. « Il faut qu’on parte », murmura-t-il, effrayé. "
        "Les vagues se brisaient violemment contre les rochers noirs en contrebas. "
        "La nuit tombait inexorablement sur le royaume abandonné."
    )
    report = analyze(text, language="fr")

    assert isinstance(report, TextReport)
    assert report.language == "fr"
    assert report.execution_time_seconds > 0.0

    assert report.volume is not None
    assert report.rhythm is not None
    assert report.style is not None
    assert report.vocabulary is not None
    assert report.readability is not None
    assert report.repetition is not None
    assert report.dialogue is not None

    data = report.to_dict()
    assert isinstance(data, dict)
    assert data["volume"]["word_count"] == report.volume.word_count


def test_invalid_metric_name_raises_value_error_with_analyzer() -> None:
    """Check passing unknown metric name raises ValueError using analyzer function."""
    text = "This is a test sentence."

    with pytest.raises(ValueError, match=r"Invalid metric\(s\): \['unknown_metric'\]"):
        analyze(text, language="en", metrics=["volume", "unknown_metric"])  # type: ignore


def test_analyze_english_test_with_parsed_doc(nlp: Language) -> None:
    """Check complete analysis English pipeline using a pre-parsed spaCy Doc."""
    text = (
        "Fog enveloped the cliff. “We have to leave,” he whispered fearfully. "
        "The waves crashed violently against the black rocks below. "
        "Night was falling inexorably over the abandoned kingdom."
    )
    doc = nlp(text)
    report = analyze(text, doc=doc, language="en")

    assert isinstance(report, TextReport)
    assert report.language == "en"
    assert report.execution_time_seconds > 0.0

    assert report.volume is not None
    assert report.rhythm is not None
    assert report.style is not None
    assert report.vocabulary is not None
    assert report.readability is not None
    assert report.repetition is not None
    assert report.dialogue is not None

    data = report.to_dict()
    assert isinstance(data, dict)
    assert data["volume"]["word_count"] == report.volume.word_count


def test_analyze_test_with_empty_metrics_list() -> None:
    """Check complete analysis English pipeline with an empty metrics list."""
    text = (
        "Fog enveloped the cliff. “We have to leave,” he whispered fearfully. "
        "The waves crashed violently against the black rocks below. "
        "Night was falling inexorably over the abandoned kingdom."
    )
    report = analyze(text, language="en", metrics=[])

    assert isinstance(report, TextReport)
    assert report.language == "en"
    assert report.execution_time_seconds > 0.0

    assert report.volume is not None
    assert report.rhythm is not None
    assert report.style is not None
    assert report.vocabulary is not None
    assert report.readability is not None
    assert report.repetition is not None

    assert report.volume.word_count > 0
    assert report.rhythm.avg_sentence_length > 0
    assert report.style.noun_ratio >= 0.0
    assert report.vocabulary.unique_word_count > 0
    assert report.readability.flesch_reading_ease > 0.0
    assert report.repetition.lexical_word_count > 0


def test_invalid_metric_string_name_raises_value_error() -> None:
    """Check passing unknown metric name raises ValueError using analyzer function."""
    text = "This is a test sentence."

    with pytest.raises(ValueError, match=r"Invalid metrics argument: 'volume'"):
        analyze(text, language="en", metrics="volume")  # type: ignore


def test_analyze_english_test_with_custom_metrics(nlp: Language) -> None:
    """Check complete analysis English pipeline using custom parameters: short_threshold, long_threshold, use_lemmas."""
    text = (
        "Fog enveloped the cliff. “We have to leave,” he whispered fearfully. "
        "The waves crashed violently against the black rocks below. "
        "Night was falling inexorably over the abandoned kingdom."
    )
    doc = nlp(text)
    report = analyze(
        text, doc=doc, language="en", short_threshold=5, long_threshold=50, use_lemmas=False, repetition_window_size=10
    )

    assert report.rhythm is not None
    assert report.vocabulary is not None
    assert report.repetition is not None

    assert report.rhythm.short_sentence_ratio == 0.25
    assert report.rhythm.long_sentence_ratio == 0.0
    assert report.vocabulary.unique_word_count > 0
    assert report.repetition.lexical_word_count > 0
