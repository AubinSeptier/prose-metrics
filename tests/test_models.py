"""Tests for prose_metrics data models."""

import dataclasses

import pytest

from prose_metrics.models import (
    DialogueMetrics,
    ReadabilityMetrics,
    RepetitionMetrics,
    RhythmMetrics,
    StyleMetrics,
    TextReport,
    VocabularyMetrics,
    VolumeMetrics,
)


@pytest.fixture
def sample_report() -> TextReport:
    """Create a sample TextReport fixture."""
    return TextReport(
        language="en",
        execution_time_seconds=0.042,
        volume=VolumeMetrics(
            character_count=1200,
            character_count_no_spaces=1000,
            word_count=200,
            sentence_count=10,
            paragraph_count=3,
            dialogue_word_count=50,
            narrative_word_count=150,
            dialogue_ratio=0.25,
        ),
        rhythm=RhythmMetrics(
            avg_sentence_length=20.0,
            sentence_length_variance=16.0,
            sentence_length_std_dev=4.0,
            short_sentence_ratio=0.1,
            long_sentence_ratio=0.2,
            punctuation_distribution={",": 12, ".": 10},
            starter_category_distribution={
                "pronoun": 5,
                "noun_phrase": 3,
                "adverb": 2,
                "preposition": 0,
                "conjunction": 0,
                "verb": 2,
                "adjective": 2,
                "other": 1,
            },
            starter_entropy=0.85,
            pronoun_starter_ratio=0.5,
            max_consecutive_starter_run=3,
        ),
        style=StyleMetrics(
            noun_ratio=0.28,
            verb_ratio=0.18,
            adjective_ratio=0.10,
            adverb_ratio=0.05,
            pronoun_ratio=0.08,
            adverbs_manner_count=3,
            pos_distribution={"NOUN": 56, "VERB": 36},
        ),
        vocabulary=VocabularyMetrics(
            unique_word_count=120,
            ttr=0.60,
            mattr=0.62,
            mattr_window_size=50,
            hapax_count=80,
            hapax_ratio=0.667,
            yule_k=0.45,
            maas_a2=0.003,
            msttr=0.58,
            msttr_segment_size=50,
        ),
        readability=ReadabilityMetrics(
            flesch_reading_ease=75.5,
            flesch_kincaid_grade=6.2,
            gunning_fog=8.4,
            estimated_reading_time_minutes=1.0,
        ),
        repetition=RepetitionMetrics(
            repetition_density=0.15,
            close_repetition_count=5,
            lexical_word_count=15,
            window_size=50,
        ),
        dialogue=DialogueMetrics(
            dialogue_verb_count=20,
            neutral_dialogue_verb_count=15,
            expressive_dialogue_verb_count=5,
            neutral_dialogue_verb_ratio=0.75,
        ),
    )


def test_text_report_immutability(sample_report: TextReport) -> None:
    """Check that dataclasses are frozen and immutable."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(sample_report, "language", "fr")


def test_text_report_to_dict(sample_report: TextReport) -> None:
    """Check that to_dict exports valid nested dict structure."""
    data = sample_report.to_dict()
    assert isinstance(data, dict)
    assert data["language"] == "en"
    assert data["volume"]["word_count"] == 200
    assert data["vocabulary"]["mattr"] == 0.62
    assert data["repetition"]["repetition_density"] == 0.15
    assert "punctuation_distribution" in data["rhythm"]
    assert "neutral_dialogue_verb_ratio" in data["dialogue"]


def test_text_report_hashable(sample_report: TextReport) -> None:
    """Check that dataclasses are hashable."""
    assert isinstance(hash(sample_report), int)
