"""Unit tests for vocabulary and lexical richness metrics."""

import pytest
from spacy.language import Language

from prose_metrics.metrics.vocabulary import compute_vocabulary_metrics
from prose_metrics.nlp.pipeline import SpacyPipelineManager


@pytest.fixture(scope="module")
def nlp() -> Language:
    """Provide a cached spaCy pipeline for tests."""
    return SpacyPipelineManager().get_pipeline(language="en")


def test_empty_text_vocabulary(nlp: Language) -> None:
    """Check empty text produces zeroed vocabulary metrics."""
    text = "   \n\n  "
    doc = nlp(text)
    metrics = compute_vocabulary_metrics(doc)

    assert metrics.unique_word_count == 0
    assert metrics.ttr == 0.0
    assert metrics.mattr == 0.0
    assert metrics.hapax_count == 0
    assert metrics.hapax_ratio == 0.0


def test_repetitive_text(nlp: Language) -> None:
    """Check metrics on entirely repetitive vocabulary."""
    text = "sun sun sun sun sun"
    doc = nlp(text)
    metrics = compute_vocabulary_metrics(doc, mattr_window_size=3)

    assert metrics.mattr == round(1 / 3, 4)
    assert metrics.hapax_count == 0
    assert metrics.hapax_ratio == 0.0


def test_all_unique_words_hapax(nlp: Language) -> None:
    """Check that distinct words yield 100% Hapax ratio and 1.0 TTR."""
    text = "The black cat looks at a little gray mouse."
    doc = nlp(text)
    metrics = compute_vocabulary_metrics(doc)

    assert metrics.ttr == 1.0
    assert metrics.hapax_count == metrics.unique_word_count
    assert metrics.hapax_ratio == 1.0


def test_mattr_window_size(nlp: Language) -> None:
    """Check fallback to standard TTR when text is shorter than window."""
    text = "A short text with a few different words."
    doc = nlp(text)
    metrics = compute_vocabulary_metrics(doc, mattr_window_size=100)

    assert metrics.mattr == metrics.ttr


def test_mattr_sliding_window_computation(nlp: Language) -> None:
    """Check sliding window MATTR on synthetic long word sequence."""
    words = [f"word{i}" for i in range(20)] + ["word0"] * 20
    text = " ".join(words)
    doc = nlp(text)
    metrics = compute_vocabulary_metrics(doc, mattr_window_size=10)

    assert 0.0 < metrics.mattr < 1.0
    assert metrics.unique_word_count == 20
    assert metrics.mattr_window_size == 10


def test_punctuation_only_text(nlp: Language) -> None:
    """Check punctuation-only text produces zeroed vocabulary metrics."""
    text = "?!...—"
    doc = nlp(text)
    metrics = compute_vocabulary_metrics(doc)

    assert metrics.unique_word_count == 0
    assert metrics.hapax_ratio == 0.0


def test_invalid_mattr_window_size(nlp: Language) -> None:
    """Check that an invalid MATTR window size raises ValueError."""
    text = "Hello world!"
    doc = nlp(text)
    with pytest.raises(ValueError, match="mattr_window_size must be >= 1"):
        compute_vocabulary_metrics(doc, mattr_window_size=0)
