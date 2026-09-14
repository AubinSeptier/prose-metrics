"""Integration and performance tests for TextAnalyzer."""

from typing import Iterator

import pytest
from spacy.language import Language

from prose_metrics import TextAnalyzer, analyze, pipe
from prose_metrics.models import TextReport
from prose_metrics.nlp.pipeline import SpacyPipelineManager


def _get_word_counts(reports: list[TextReport]) -> list[int]:
    """Extract word counts from reports, asserting volume metrics are present."""
    word_counts: list[int] = []
    for report in reports:
        assert report.volume is not None
        word_counts.append(report.volume.word_count)
    return word_counts


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


def test_pipe_matches_analyze_results() -> None:
    """Check pipe() produces the same metrics as analyze() for each text."""
    texts = [
        "The fog rolled in over the harbor at dawn.",
        "“We should go now,” she said quietly, glancing at the door.",
        "He ran and ran until his lungs burned with cold air.",
    ]
    analyzer = TextAnalyzer(language="en")
    reports = list(analyzer.pipe(texts))

    assert len(reports) == len(texts)
    for text, report in zip(texts, reports, strict=True):
        expected = analyzer.analyze(text)
        assert report.language == expected.language
        assert report.volume == expected.volume
        assert report.rhythm == expected.rhythm
        assert report.style == expected.style
        assert report.vocabulary == expected.vocabulary
        assert report.readability == expected.readability
        assert report.repetition == expected.repetition
        assert report.dialogue == expected.dialogue


def test_pipe_preserves_input_order() -> None:
    """Check reports are yielded in the same order as the input texts."""
    texts = ["The cat sleeps.", "A very big brown dog.", "Short."]
    analyzer = TextAnalyzer(language="en")
    word_counts = _get_word_counts(list(analyzer.pipe(texts)))
    assert word_counts == [3, 5, 1]


def test_pipe_does_not_consume_iterable_before_first_next() -> None:
    """Check parsing is lazy and generator input is correctly paired."""
    consumed: list[str] = []

    def tracking_texts() -> Iterator[str]:
        for text in ["The cat sleeps.", "A very big brown dog."]:
            consumed.append(text)
            yield text

    analyzer = TextAnalyzer(language="en")
    reports = analyzer.pipe(tracking_texts())
    assert consumed == []

    assert _get_word_counts(list(reports)) == [3, 5]
    assert len(consumed) == 2


def test_pipe_with_empty_iterable_returns_no_reports() -> None:
    """Check piping an empty iterable yields no reports."""
    assert list(TextAnalyzer(language="en").pipe([])) == []


def test_pipe_invalid_metrics_raise_before_consumption() -> None:
    """Check metric validation is eager and leaves the input iterable untouched."""
    consumed: list[str] = []

    def tracking_texts() -> Iterator[str]:
        consumed.append("consumed")
        yield "Some text."

    analyzer = TextAnalyzer(language="en")
    with pytest.raises(ValueError, match=r"Invalid metric\(s\): \['unknown'\]"):
        analyzer.pipe(tracking_texts(), metrics=["unknown"])  # type: ignore

    assert consumed == []


def test_pipe_invalid_pipe_parameters_raise_value_error() -> None:
    """Check batch_size and n_process below 1 raise ValueError."""
    analyzer = TextAnalyzer(language="en")

    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        analyzer.pipe(["Some text."], batch_size=0)

    with pytest.raises(ValueError, match="n_process must be at least 1"):
        analyzer.pipe(["Some text."], n_process=0)


def test_pipe_with_selective_metrics() -> None:
    """Check pipe() respects a subset of metrics."""
    analyzer = TextAnalyzer(language="en")
    reports = list(analyzer.pipe(["A simple test sentence."], metrics=["volume"]))

    assert reports[0].volume is not None
    assert reports[0].rhythm is None
    assert reports[0].style is None


@pytest.mark.filterwarnings(r"ignore:This process \(pid=\d+\) is multi-threaded:DeprecationWarning")
def test_pipe_with_multiprocessing() -> None:
    """Check pipe() with n_process=2 yields correct reports in order."""
    texts = ["The cat sleeps quietly.", "A dog runs very fast."]
    analyzer = TextAnalyzer(language="en")
    reports = list(analyzer.pipe(texts, batch_size=1, n_process=2))

    assert _get_word_counts(reports) == [4, 5]


def test_module_level_pipe_function() -> None:
    """Check the module-level pipe() convenience function."""
    texts = ["The cat sleeps quietly.", "A dog runs very fast."]
    reports = list(pipe(texts, language="en"))

    assert len(reports) == 2
    assert all(isinstance(report, TextReport) for report in reports)
    assert _get_word_counts(reports) == [4, 5]


def test_module_level_pipe_invalid_metrics_raise_value_error() -> None:
    """Check the module-level pipe() validates metrics eagerly."""
    with pytest.raises(ValueError, match=r"Invalid metric\(s\): \['unknown'\]"):
        pipe(["Some text."], language="en", metrics=["unknown"])  # type: ignore
