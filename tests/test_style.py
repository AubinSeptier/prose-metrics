"""Unit tests for stylistic and grammatical metrics."""

import pytest
from spacy.language import Language

from prose_metrics.metrics.style import compute_style_metrics
from prose_metrics.nlp.pipeline import SpacyPipelineManager


@pytest.fixture(scope="module")
def nlp() -> Language:
    """Provide a cached spaCy pipeline for tests."""
    return SpacyPipelineManager().get_pipeline(language="en")


@pytest.fixture(scope="module")
def nlp_fr() -> Language:
    """Provide a cached spaCy pipeline for French tests."""
    return SpacyPipelineManager().get_pipeline(language="fr")


def test_empty_text_style(nlp: Language) -> None:
    """Check metrics on empty and whitespace-only text."""
    text = "   \n\n  "
    doc = nlp(text)
    metrics = compute_style_metrics(doc)

    assert metrics.noun_ratio == 0.0
    assert metrics.verb_ratio == 0.0
    assert metrics.adjective_ratio == 0.0
    assert metrics.adverb_ratio == 0.0
    assert metrics.pronoun_ratio == 0.0
    assert metrics.adverbs_manner_count == 0
    assert metrics.pos_distribution == {}


def test_english_manner_adverbs_and_false_positives(nlp: Language) -> None:
    """Check detection of manner adverbs while ignoring nouns ending in -ly."""
    text = "The lovely family walked quickly and quietly into the house."
    doc = nlp(text)
    metrics = compute_style_metrics(doc)

    assert metrics.adverbs_manner_count == 2
    assert metrics.adverb_ratio > 0.0
    assert metrics.noun_ratio > 0.0
    assert metrics.adjective_ratio > 0.0
    assert metrics.verb_ratio > 0.0
    assert "NOUN" in metrics.pos_distribution


def test_pos_ratios_consistency(nlp: Language) -> None:
    """Check all ratios stay strictly within [0.0, 1.0] and metrics consistency."""
    text = "The big black cat was sleeping peacefully on the red sofa."
    doc = nlp(text)
    metrics = compute_style_metrics(doc)

    dist = metrics.pos_distribution
    total_words = sum(dist.values())
    assert total_words == 11

    for ratio in (
        metrics.noun_ratio,
        metrics.verb_ratio,
        metrics.adjective_ratio,
        metrics.adverb_ratio,
        metrics.pronoun_ratio,
    ):
        assert 0.0 <= ratio <= 1.0

    assert metrics.noun_ratio == round((dist.get("NOUN", 0) + dist.get("PROPN", 0)) / total_words, 4)
    assert metrics.verb_ratio == round((dist.get("VERB", 0) + dist.get("AUX", 0)) / total_words, 4)
    assert metrics.adjective_ratio == round(dist.get("ADJ", 0) / total_words, 4)
    assert metrics.adverb_ratio == round(dist.get("ADV", 0) / total_words, 4)
    assert metrics.pronoun_ratio == round((dist.get("PRON", 0)) / total_words, 4)
    assert metrics.adverbs_manner_count <= dist.get("ADV", 0)


def test_pronoun_ratio(nlp: Language) -> None:
    """Check pronoun ratio is computed from PRON tokens."""
    text = "She told him the story."
    doc = nlp(text)
    metrics = compute_style_metrics(doc)

    assert metrics.pronoun_ratio == 0.4
    assert metrics.pos_distribution["PRON"] == 2


def test_irregular_manner_adverbs(nlp: Language) -> None:
    """Check irregular manner adverbs without the -ly suffix are counted."""
    text = "She sings well. He runs fast."
    doc = nlp(text)
    metrics = compute_style_metrics(doc)

    assert metrics.adverbs_manner_count == 2


def test_english_non_manner_adverbs(nlp: Language) -> None:
    """Check -ly adverbs of time and intensity are excluded."""
    text = "He hardly walked. I finally arrived home."
    doc = nlp(text)
    metrics = compute_style_metrics(doc)

    assert metrics.adverbs_manner_count == 0


def test_adjective_modifying_adverb_excluded(nlp: Language) -> None:
    """Check adverbs modifying adjectives are not counted as manner adverbs."""
    text = "The very big cat slept."
    doc = nlp(text)
    metrics = compute_style_metrics(doc)

    assert metrics.adverbs_manner_count == 0


def test_flat_adverb_tagged_adjective(nlp: Language) -> None:
    """Check irregular flat adverbs tagged as adjectives are still counted."""
    text = "She did it wrong."
    doc = nlp(text)
    metrics = compute_style_metrics(doc)

    assert metrics.adverbs_manner_count == 1


def test_french_manner_adverbs_and_false_positives(nlp_fr: Language) -> None:
    """Check detection of French manner adverbs while ignoring nouns ending in -ly."""
    text = "Cette charmante famille est entrée rapidement et silencieusement dans la maison."
    doc = nlp_fr(text)
    metrics = compute_style_metrics(doc)

    assert metrics.adverbs_manner_count == 2
    assert metrics.adverb_ratio > 0.0
    assert metrics.noun_ratio > 0.0
    assert metrics.adjective_ratio > 0.0
    assert metrics.verb_ratio > 0.0
    assert "NOUN" in metrics.pos_distribution


def test_french_non_manner_adverbs(nlp_fr: Language) -> None:
    """Check -ment adverbs of time and intensity are excluded."""
    text = "J'ai énormément marché. Je suis enfin arrivé à la maison."
    doc = nlp_fr(text)
    metrics = compute_style_metrics(doc)

    assert metrics.adverbs_manner_count == 0


def test_french_irregular_manner_adverbs(nlp_fr: Language) -> None:
    """Check irregular French manner adverbs without the -ment suffix are counted."""
    text = "Elle chante bien. Il marche vite."
    doc = nlp_fr(text)
    metrics = compute_style_metrics(doc)

    assert metrics.adverbs_manner_count == 2
