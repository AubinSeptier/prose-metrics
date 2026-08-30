"""Unit tests for repetition metrics."""

import pytest
from spacy.language import Language

from prose_metrics.metrics.repetition import compute_repetition_metrics
from prose_metrics.nlp.pipeline import SpacyPipelineManager


@pytest.fixture(scope="module")
def nlp() -> Language:
    """Provide a cached spaCy pipeline for tests."""
    return SpacyPipelineManager().get_pipeline(language="en")


@pytest.fixture(scope="module")
def nlp_fr() -> Language:
    """Provide a cached spaCy pipeline for French tests."""
    return SpacyPipelineManager().get_pipeline(language="fr")


def test_empty_text_repetition(nlp: Language) -> None:
    """Check empty test produces zeroed repetition metrics."""
    text = "   \n\n  "
    doc = nlp(text)
    metrics = compute_repetition_metrics(doc)

    assert metrics.lexical_word_count == 0
    assert metrics.close_repetition_count == 0
    assert metrics.repetition_density == 0.0


def test_close_repetition_detected(nlp: Language) -> None:
    """Check exact metrics on a sentence with one close repetition."""
    text = "The dog barked loudly. The cat ran fast. The dog slept peacefully."
    doc = nlp(text)
    metrics = compute_repetition_metrics(doc)

    assert metrics.lexical_word_count == 9
    assert metrics.close_repetition_count == 1
    assert metrics.repetition_density == 0.1111
    assert metrics.window_size == 50


def test_no_repetition(nlp: Language) -> None:
    """Check that distinct content words yield zero repetitions."""
    text = "The black cat looks at a little gray mouse."
    doc = nlp(text)
    metrics = compute_repetition_metrics(doc)

    assert metrics.lexical_word_count == 6
    assert metrics.close_repetition_count == 0
    assert metrics.repetition_density == 0.0


def test_window_size_boundary(nlp: Language) -> None:
    """Check that a repetition counts at distance == window_size but not at window_size + 1."""
    text = "The dog chased the big cat. The dog slept."
    doc = nlp(text)

    within = compute_repetition_metrics(doc, window_size=4)
    beyond = compute_repetition_metrics(doc, window_size=3)

    assert within.lexical_word_count == 6
    assert within.close_repetition_count == 1
    assert within.repetition_density == 0.1667
    assert beyond.close_repetition_count == 0
    assert beyond.window_size == 3


def test_multiple_recurrences(nlp: Language) -> None:
    """Check that only second and subsequent occurrences are counted."""
    text = "The dog barked, the dog ran, the dog slept, the dog ate, the dog played."
    doc = nlp(text)
    metrics = compute_repetition_metrics(doc)

    assert metrics.lexical_word_count == 10
    assert metrics.close_repetition_count == 4
    assert metrics.repetition_density == 0.4


def test_lemmas_group_inflections(nlp: Language) -> None:
    """Check that lemma mode groups inflected forms, raw mode does not."""
    text = "The cats are cute. That cat sleeps."
    doc = nlp(text)

    lemma_metrics = compute_repetition_metrics(doc, use_lemmas=True)
    raw_metrics = compute_repetition_metrics(doc, use_lemmas=False)

    assert lemma_metrics.lexical_word_count == 4
    assert lemma_metrics.close_repetition_count == 1
    assert lemma_metrics.repetition_density == 0.25
    assert raw_metrics.close_repetition_count == 0
    assert raw_metrics.repetition_density == 0.0


def test_proper_nouns_excluded(nlp: Language) -> None:
    """Check that repeated character names (PROPN) are not flagged as repetitions."""
    text = "Alice smiled. Alice laughed."
    doc = nlp(text)
    metrics = compute_repetition_metrics(doc)

    assert metrics.lexical_word_count == 2
    assert metrics.close_repetition_count == 0
    assert metrics.repetition_density == 0.0


def test_french_repetition(nlp_fr: Language) -> None:
    """Check close repetition detection on a French text."""
    text = "Le chat dort. Le chien court. Les chats mangent."
    doc = nlp_fr(text)
    metrics = compute_repetition_metrics(doc)

    assert metrics.lexical_word_count == 6
    assert metrics.close_repetition_count == 1
    assert metrics.repetition_density == 0.1667


def test_punctuation_only_text(nlp: Language) -> None:
    """Check punctuation-only text produces zeroed repetition metrics."""
    text = "?!...-"
    doc = nlp(text)
    metrics = compute_repetition_metrics(doc)

    assert metrics.lexical_word_count == 0
    assert metrics.close_repetition_count == 0
    assert metrics.repetition_density == 0.0


def test_invalid_window_size_raises_value_error(nlp: Language) -> None:
    """Check that a invalid window size raises ValueError."""
    text = "The dog barked. The cat ran."
    doc = nlp(text)

    with pytest.raises(ValueError, match="window_size must be >= 1"):
        compute_repetition_metrics(doc, window_size=0)
