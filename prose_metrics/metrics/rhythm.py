"""Rhythm metrics for analyzing the rhythmic patterns in text."""

import statistics
from collections import Counter
from typing import Final

from spacy.tokens import Doc

from prose_metrics.models.report import RhythmMetrics

DEFAULT_SHORT_THRESHOLD: Final[int] = 10
DEFAULT_LONG_THRESHOLD: Final[int] = 30

TRACKED_PUNCTUATION: Final[tuple[str, ...]] = (
    ",",
    ".",
    "!",
    "?",
    ";",
    ":",
    "—",
    "…",
)


def compute_rhythm_metrics(
    doc: Doc,
    short_threshold: int = DEFAULT_SHORT_THRESHOLD,
    long_threshold: int = DEFAULT_LONG_THRESHOLD,
) -> RhythmMetrics:
    """Compute sentence rhythm dispersion metrics and punctuation distribution.

    Args:
        doc (Doc): The parsed spaCy Doc instance.
        short_threshold (int): Upper word count bound for short sentences (< threshold). Defaults to 10 words.
        long_threshold (int): Lower word count bound for long sentences (> threshold). Defaults to 30 words.

    Returns:
        RhythmMetrics: An instance containing sentence length variations and punctuations counts.

    Examples:
        >>> from prose_metrics.nlp.pipeline import SpacyPipelineManager
        >>> doc = SpacyPipelineManager().get_pipeline("en")("The cat sat. It rained all day; the wind howled.")
        >>> metrics = compute_rhythm_metrics(doc)
        >>> metrics.avg_sentence_length
        5.0
        >>> metrics.short_sentence_ratio
        1.0
        >>> metrics.punctuation_distribution["."]
        1
        >>> metrics.punctuation_distribution["…"]
        1
        >>> compute_rhythm_metrics(doc, short_threshold=4).short_sentence_ratio
        0.5
    """
    sentence_word_lengths: list[int] = []

    # Compute word count per active sentence
    for sent in doc.sents:
        word_count = sum(1 for tok in sent if not (tok.is_punct or tok.is_space))
        if word_count > 0:
            sentence_word_lengths.append(word_count)

    total_sentences = len(sentence_word_lengths)

    if total_sentences == 0:
        avg_sentence_length = 0.0
        sentence_length_variance = 0.0
        sentence_length_std_dev = 0.0
        short_sentence_ratio = 0.0
        long_sentence_ratio = 0.0
    else:
        avg_sentence_length = round(statistics.fmean(sentence_word_lengths), 2)
        sentence_length_variance = round(statistics.pvariance(sentence_word_lengths), 2)
        sentence_length_std_dev = round(statistics.pstdev(sentence_word_lengths), 2)

        short_count = sum(1 for length in sentence_word_lengths if length < short_threshold)
        long_count = sum(1 for length in sentence_word_lengths if length > long_threshold)

        short_sentence_ratio = round(short_count / total_sentences, 4)
        long_sentence_ratio = round(long_count / total_sentences, 4)

    # Punctuation distribution
    raw_punct_counter: Counter[str] = Counter()
    for token in doc:
        if token.is_punct:
            punct = token.text
            # Normalize triple-dot tokens to standard ellipsis character
            if punct == "...":
                punct = "…"
            raw_punct_counter[punct] += 1

    punctuation_distribution: dict[str, int] = {punct: raw_punct_counter.get(punct, 0) for punct in TRACKED_PUNCTUATION}

    return RhythmMetrics(
        avg_sentence_length=avg_sentence_length,
        sentence_length_variance=sentence_length_variance,
        sentence_length_std_dev=sentence_length_std_dev,
        short_sentence_ratio=short_sentence_ratio,
        long_sentence_ratio=long_sentence_ratio,
        punctuation_distribution=punctuation_distribution,
    )
