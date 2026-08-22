"""Unit tests for volumetric metrics."""

import pytest
from spacy.language import Language

from prose_metrics.metrics.volume import compute_volume_metrics
from prose_metrics.nlp.pipeline import SpacyPipelineManager


@pytest.fixture(scope="module")
def nlp() -> Language:
    """Provide a cached spaCy pipeline for tests."""
    return SpacyPipelineManager().get_pipeline(language="en")


def test_empty_text(nlp: Language) -> None:
    """Check metrics on empty and whitespace-only text."""
    text = "   \n\n  "
    doc = nlp(text)
    metrics = compute_volume_metrics(text, doc)

    assert metrics.character_count == 7
    assert metrics.character_count_no_spaces == 0
    assert metrics.word_count == 0
    assert metrics.sentence_count == 0
    assert metrics.paragraph_count == 0
    assert metrics.dialogue_word_count == 0
    assert metrics.narrative_word_count == 0
    assert metrics.dialogue_ratio == 0.0


def test_french_guillemets_dialogue(nlp: Language) -> None:
    """Check dialogue detection with French guillemets (« »)."""
    text = "The knight stepped forward. « Halt! » shouted the guard."
    doc = nlp(text)
    metrics = compute_volume_metrics(text, doc)

    assert metrics.sentence_count == 2
    assert metrics.paragraph_count == 1
    assert metrics.dialogue_word_count == 1
    assert metrics.narrative_word_count == metrics.word_count - 1
    assert metrics.dialogue_ratio == round(1 / metrics.word_count, 4)


def test_dash_dialogue(nlp: Language) -> None:
    """Check dialogue detection with dialogue dashes."""
    text = "He opened the door.\n— Good morning, everyone.\nThe room was silent."
    doc = nlp(text)
    metrics = compute_volume_metrics(text, doc)
    assert metrics.paragraph_count == 3
    assert metrics.dialogue_word_count == 3
    assert metrics.narrative_word_count == metrics.word_count - 3
    assert metrics.dialogue_word_count + metrics.narrative_word_count == metrics.word_count


def test_character_counts(nlp: Language) -> None:
    """Check total and non-whitespace character counting."""
    text = "One word. Two words."
    doc = nlp(text)
    metrics = compute_volume_metrics(text, doc)

    assert metrics.character_count == len(text)
    assert metrics.character_count_no_spaces == len("Oneword.Twowords.")
    assert metrics.word_count == 4
