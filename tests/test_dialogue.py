"""Unit tests for dialogue metrics."""

import pytest
from spacy.language import Language

from prose_metrics.metrics.dialogue import compute_dialogue_metrics
from prose_metrics.nlp.pipeline import SpacyPipelineManager


@pytest.fixture(scope="module")
def nlp() -> Language:
    """Provide a cached spaCy pipeline for tests."""
    return SpacyPipelineManager().get_pipeline(language="en")


@pytest.fixture(scope="module")
def nlp_fr() -> Language:
    """Provide a cached spaCy pipeline for French tests."""
    return SpacyPipelineManager().get_pipeline(language="fr")


def test_empty_text_dialogue(nlp: Language) -> None:
    """Check empty text produces zeroed dialogue metrics."""
    text = "   \n\n  "
    doc = nlp(text)
    metrics = compute_dialogue_metrics(text, doc)

    assert metrics.dialogue_verb_count == 0
    assert metrics.neutral_dialogue_verb_count == 0
    assert metrics.expressive_dialogue_verb_count == 0
    assert metrics.neutral_dialogue_verb_ratio == 0.0


def test_punctuation_only_dialogue(nlp: Language) -> None:
    """Check that punctuation-only text produces zeroed dialogue metrics."""
    text = "“”‘’.,!?;:()[]{}"
    doc = nlp(text)
    metrics = compute_dialogue_metrics(text, doc)

    assert metrics.dialogue_verb_count == 0
    assert metrics.neutral_dialogue_verb_count == 0
    assert metrics.expressive_dialogue_verb_count == 0
    assert metrics.neutral_dialogue_verb_ratio == 0.0


def test_dialogue_metrics_basic(nlp: Language) -> None:
    """Check basic dialogue metrics calculation."""
    text = "“Run now,” she whispered. “Wait,” he said."
    doc = nlp(text)
    metrics = compute_dialogue_metrics(text, doc)

    assert metrics.dialogue_verb_count == 2
    assert metrics.neutral_dialogue_verb_count == 1
    assert metrics.expressive_dialogue_verb_count == 1
    assert metrics.neutral_dialogue_verb_ratio == 0.5


def test_expressive_tags_after_terminal_punctuation(nlp: Language) -> None:
    """Check tags are found after dialogue-final '!' or '?' despite spaCy sentence splits."""
    text = "“Stop!” she shouted. “Now.” he whispered."
    doc = nlp(text)
    metrics = compute_dialogue_metrics(text, doc)

    assert metrics.dialogue_verb_count == 2
    assert metrics.neutral_dialogue_verb_count == 0
    assert metrics.expressive_dialogue_verb_count == 2
    assert metrics.neutral_dialogue_verb_ratio == 0.0


def test_tag_deduplication_across_spans(nlp: Language) -> None:
    """Check a tag splitting two dialogue spans is counted only once."""
    text = "“Yes,” she said, “we can.”"
    doc = nlp(text)
    metrics = compute_dialogue_metrics(text, doc)

    assert metrics.dialogue_verb_count == 1
    assert metrics.neutral_dialogue_verb_count == 1
    assert metrics.neutral_dialogue_verb_ratio == 1.0


def test_pre_dialogue_tag(nlp: Language) -> None:
    """Check tags placed before the dialogue span are detected."""
    text = "John said: “Run.”"
    doc = nlp(text)
    metrics = compute_dialogue_metrics(text, doc)

    assert metrics.dialogue_verb_count == 1
    assert metrics.neutral_dialogue_verb_count == 1


def test_dialogue_without_tag(nlp: Language) -> None:
    """Check dialogue lines without a reporting verb yield zero counts."""
    text = "“Hello.” The door closed."
    doc = nlp(text)
    metrics = compute_dialogue_metrics(text, doc)

    assert metrics.dialogue_verb_count == 0
    assert metrics.neutral_dialogue_verb_ratio == 0.0


def test_sentence_guard(nlp: Language) -> None:
    """Check speech verbs in the sentence following untagged dialogue are not attributed."""
    text = "“Hello.” The door closed quietly. He said nothing."
    doc = nlp(text)
    metrics = compute_dialogue_metrics(text, doc)

    assert metrics.dialogue_verb_count == 0


def test_distance_guard(nlp: Language) -> None:
    """Check speech verbs beyond the maximum tag distance are not attributed."""
    text = "“Hello.” She looked at the old wooden door standing right there and said nothing."
    doc = nlp(text)
    metrics = compute_dialogue_metrics(text, doc)

    assert metrics.dialogue_verb_count == 0


def test_straight_quotes_dialogue(nlp: Language) -> None:
    """Check tags are detected with straight double quotes convention."""
    text = '"Run," she said.'
    doc = nlp(text)
    metrics = compute_dialogue_metrics(text, doc)

    assert metrics.dialogue_verb_count == 1
    assert metrics.neutral_dialogue_verb_count == 1
    assert metrics.neutral_dialogue_verb_ratio == 1.0


def test_neutral_ratio_rounding(nlp: Language) -> None:
    """Check neutral ratio rounding with a mixed set of three tags."""
    text = "“Wait,” he said. “Stop!” she shouted. “Now.” he whispered."
    doc = nlp(text)
    metrics = compute_dialogue_metrics(text, doc)

    assert metrics.dialogue_verb_count == 3
    assert metrics.neutral_dialogue_verb_count == 1
    assert metrics.expressive_dialogue_verb_count == 2
    assert metrics.neutral_dialogue_verb_ratio == 0.3333


def test_french_incise_tags(nlp_fr: Language) -> None:
    """Check neutral and expressive tag detection in French incises."""
    text = "« Il faut qu’on parte », murmura-t-il. « Reste », dit-elle."
    doc = nlp_fr(text)
    metrics = compute_dialogue_metrics(text, doc)

    assert metrics.dialogue_verb_count == 2
    assert metrics.neutral_dialogue_verb_count == 1
    assert metrics.expressive_dialogue_verb_count == 1
    assert metrics.neutral_dialogue_verb_ratio == 0.5


def test_french_expressive_only(nlp_fr: Language) -> None:
    """Check a French expressive tag yields a zero neutral ratio."""
    text = "« Viens », murmura-t-elle doucement."
    doc = nlp_fr(text)
    metrics = compute_dialogue_metrics(text, doc)

    assert metrics.dialogue_verb_count == 1
    assert metrics.expressive_dialogue_verb_count == 1
    assert metrics.neutral_dialogue_verb_ratio == 0.0


def test_dash_dialogue_tags_not_detected(nlp_fr: Language) -> None:
    """Check the documented limitation: dash-style dialogue incises are inside the span."""
    text = "— Je viens, dit-il.\n— Bien."
    doc = nlp_fr(text)
    metrics = compute_dialogue_metrics(text, doc)

    assert metrics.dialogue_verb_count == 0
    assert metrics.neutral_dialogue_verb_ratio == 0.0


def test_unsupported_language_zeroed() -> None:
    """Check languages without a speech-verb lexicon yield zeroed metrics."""
    import spacy

    nlp_de = spacy.blank("hr")
    text = '"Dođi ovamo odmah," rekao je glasno.'
    metrics = compute_dialogue_metrics(text, nlp_de(text))

    assert metrics.dialogue_verb_count == 0
    assert metrics.neutral_dialogue_verb_count == 0
    assert metrics.expressive_dialogue_verb_count == 0
    assert metrics.neutral_dialogue_verb_ratio == 0.0
