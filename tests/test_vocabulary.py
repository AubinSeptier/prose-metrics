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
    assert metrics.yule_k == 0.0
    assert metrics.maas_a2 == 0.0
    assert metrics.msttr == 0.0


def test_repetitive_text(nlp: Language) -> None:
    """Check metrics on entirely repetitive vocabulary."""
    text = "sun sun sun sun sun"
    doc = nlp(text)
    metrics = compute_vocabulary_metrics(doc, mattr_window_size=3)

    assert metrics.mattr == round(1 / 3, 4)
    assert metrics.hapax_count == 0
    assert metrics.hapax_ratio == 0.0
    assert metrics.yule_k == 8000.0
    assert metrics.maas_a2 == 0.6213


def test_all_unique_words_hapax(nlp: Language) -> None:
    """Check that distinct words yield 100% Hapax ratio and 1.0 TTR."""
    text = "The black cat looks at a little gray mouse."
    doc = nlp(text)
    metrics = compute_vocabulary_metrics(doc)

    assert metrics.ttr == 1.0
    assert metrics.hapax_count == metrics.unique_word_count
    assert metrics.hapax_ratio == 1.0
    assert metrics.yule_k == 0.0
    assert metrics.maas_a2 == 0.0


def test_mattr_window_size_fallback(nlp: Language) -> None:
    """Check MATTR fallback to standard TTR when text is shorter than window."""
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
    assert metrics.ttr == 0.0
    assert metrics.mattr == 0.0
    assert metrics.hapax_count == 0
    assert metrics.hapax_ratio == 0.0
    assert metrics.yule_k == 0.0
    assert metrics.maas_a2 == 0.0
    assert metrics.msttr == 0.0


def test_invalid_mattr_window_size(nlp: Language) -> None:
    """Check that an invalid MATTR window size raises ValueError."""
    text = "Hello world!"
    doc = nlp(text)
    with pytest.raises(ValueError, match="mattr_window_size must be >= 1"):
        compute_vocabulary_metrics(doc, mattr_window_size=0)


def test_maas_undefined_single_token(nlp: Language) -> None:
    """Check Maas a^2 falls back to 0.0 on a single-token text (ln 1 = 0)."""
    text = "Hello."
    doc = nlp(text)
    metrics = compute_vocabulary_metrics(doc)

    assert metrics.maas_a2 == 0.0
    assert metrics.yule_k == 0.0


def test_msttr_disjoint_segments(nlp: Language) -> None:
    """Check MSTTR averages TTR over full segments and discards the partial tail."""
    words = [f"word{i}" for i in range(100)] + [f"word{i}" for i in range(50)] * 2 + ["word0"] * 50
    text = " ".join(words)
    doc = nlp(text)
    metrics = compute_vocabulary_metrics(doc, msttr_segment_size=100)

    assert metrics.msttr == 0.75
    assert metrics.msttr_segment_size == 100


def test_msttr_exact_segment_fit(nlp: Language) -> None:
    """Check no segment is discarded when the text length is an exact multiple."""
    words = [f"word{i}" for i in range(100)] + [f"word{i}" for i in range(50)] * 2
    text = " ".join(words)
    doc = nlp(text)
    metrics = compute_vocabulary_metrics(doc, msttr_segment_size=100)

    assert metrics.msttr == 0.75


def test_msttr_segment_size_fallback(nlp: Language) -> None:
    """Check MSTTR fallback to standard TTR when no full segment fits."""
    text = "A short text with a few different words."
    doc = nlp(text)
    metrics = compute_vocabulary_metrics(doc, msttr_segment_size=100)

    assert metrics.msttr == metrics.ttr


def test_invalid_msttr_segment_size(nlp: Language) -> None:
    """Check that an invalid MSTTR segment size raises ValueError."""
    text = "Hello world!"
    doc = nlp(text)
    with pytest.raises(ValueError, match="msttr_segment_size must be >= 1"):
        compute_vocabulary_metrics(doc, msttr_segment_size=0)
