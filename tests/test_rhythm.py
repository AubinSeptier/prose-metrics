"""Unit tests for rhythm and punctuation metrics."""

import pytest
from spacy.language import Language

from prose_metrics.metrics.rhythm import TRACKED_PUNCTUATION, compute_rhythm_metrics
from prose_metrics.nlp.pipeline import SpacyPipelineManager


@pytest.fixture(scope="module")
def nlp() -> Language:
    """Provide a cached spaCy pipeline for tests."""
    return SpacyPipelineManager().get_pipeline(language="en")


def test_empty_text_rhythm(nlp: Language) -> None:
    """Check metrics on empty and whitespace-only text."""
    text = "   \n\n  "
    doc = nlp(text)
    metrics = compute_rhythm_metrics(doc)

    assert metrics.avg_sentence_length == 0.0
    assert metrics.sentence_length_variance == 0.0
    assert metrics.sentence_length_std_dev == 0.0
    assert metrics.short_sentence_ratio == 0.0
    assert metrics.long_sentence_ratio == 0.0
    assert all(count == 0 for count in metrics.punctuation_distribution.values())


def test_single_sentence_variance(nlp: Language) -> None:
    """Check single sentence produces zero variance and std deviation without errors."""
    text = "The wind was blowing hard that morning on the cliff."
    doc = nlp(text)
    metrics = compute_rhythm_metrics(doc)

    assert metrics.avg_sentence_length > 0.0
    assert metrics.sentence_length_variance == 0.0
    assert metrics.sentence_length_std_dev == 0.0


def test_cadence_variation_and_ratios(nlp: Language) -> None:
    """Check short and long sentences ratios and variance calculations."""
    text = (
        "He set out alone. "
        "In the pitch-black darkness of the winter night, he walked relentlessly toward the mountaintop "
        "as gusts of icy snow lashed violently against his exhausted face and the biting cold gradually "
        "numbed his battered fingers"
    )
    doc = nlp(text)
    metrics = compute_rhythm_metrics(doc)

    assert metrics.short_sentence_ratio == 0.5
    assert metrics.long_sentence_ratio == 0.5
    assert metrics.sentence_length_variance > 100.0
    assert metrics.sentence_length_std_dev > 10.0


def test_punctuation_distribution(nlp: Language) -> None:
    """Check accurate extraction of various punctuation marks."""
    text = "Hello, he said; Really? Yes! That's how it is... the end."
    doc = nlp(text)
    metrics = compute_rhythm_metrics(doc)

    punct = metrics.punctuation_distribution
    assert punct[","] == 1
    assert punct[";"] == 1
    assert punct["?"] == 1
    assert punct["!"] == 1
    assert punct["…"] == 1
    assert punct["."] == 1


def test_threshold_boundaries_are_strict(nlp: Language) -> None:
    """Check sentences at a threshold count as neither short nor long."""
    ten_words = " ".join(f"w{i}" for i in range(1, 11)) + "."
    thirty_words = " ".join(f"w{i}" for i in range(1, 31)) + "."
    doc = nlp(ten_words + " " + thirty_words)
    metrics = compute_rhythm_metrics(doc)

    assert metrics.avg_sentence_length == 20.0
    assert metrics.short_sentence_ratio == 0.0  # 10 is not < 10
    assert metrics.long_sentence_ratio == 0.0  # 30 is not > 30


def test_custom_thresholds(nlp: Language) -> None:
    """Check short and long ratios respond to explicit threshold oevrrides."""
    text = "One two three four five six seven eight nine ten."
    doc = nlp(text)

    default_metrics = compute_rhythm_metrics(doc)
    assert default_metrics.short_sentence_ratio == 0.0  # Default threshold is 10

    custom_metrics = compute_rhythm_metrics(doc, short_threshold=11, long_threshold=30)
    assert custom_metrics.short_sentence_ratio == 1.0
    assert custom_metrics.long_sentence_ratio == 0.0


def test_em_dash_tracked_hyphen_not_tracked(nlp: Language) -> None:
    """Check em dashes are counted while word-internal hyphens are not tracked."""
    text = "He paused—then left. It was pitch-black."
    doc = nlp(text)
    metrics = compute_rhythm_metrics(doc)

    punct = metrics.punctuation_distribution
    assert punct["—"] == 1
    assert "-" not in punct
    assert punct["."] == 2


def test_ellipsis_forms_normalized(nlp: Language) -> None:
    """Check both '...' and '…' tokens land under the single '…' key."""
    text = "Wait... really? Then… silence."
    doc = nlp(text)
    metrics = compute_rhythm_metrics(doc)

    punct = metrics.punctuation_distribution
    assert punct["…"] == 2
    assert punct["."] == 1
    assert punct["?"] == 1


def test_distribution_shape_is_fixed(nlp: Language) -> None:
    """Check the distribution always contains every tracked key, even at zero."""
    text = "Only plain words."
    doc = nlp(text)
    metrics = compute_rhythm_metrics(doc)

    punct = metrics.punctuation_distribution
    assert set(punct.keys()) == set(TRACKED_PUNCTUATION)
    assert punct["."] == 1
    assert punct["!"] == 0
