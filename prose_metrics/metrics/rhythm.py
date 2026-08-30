"""Rhythm metrics for analyzing the rhythmic patterns in text."""

import math
import statistics
from collections import Counter
from typing import Final

from spacy.tokens import Doc, Span

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

_STARTER_CATEGORIES: Final[dict[str, str]] = {
    "PRON": "pronoun",
    "NOUN": "noun_phrase",
    "PROPN": "noun_phrase",
    "DET": "noun_phrase",
    "ADV": "adverb",
    "ADP": "preposition",
    "CCONJ": "conjunction",
    "SCONJ": "conjunction",
    "VERB": "verb",
    "AUX": "verb",
    "ADJ": "adjective",
}

_STARTER_CATEGORY_KEYS: Final[tuple[str, ...]] = (
    "pronoun",
    "noun_phrase",
    "adverb",
    "preposition",
    "conjunction",
    "verb",
    "adjective",
    "other",
)


def _classify_starter(sentence: Span) -> str:
    """Classify the first content token of a sentence into a starter category.

    Punctuation and whitespace tokens are skipped, so sentences opening with dialogue delimiters
    (curly quotes, guillemets, etc.) are classified on their first word.

    Args:
        sentence (Span): The spaCy sentence span.

    Returns:
        str: The starter category (e.g., "pronoun", "noun_phrase", "adverb", etc.) or "other" for unmapped POS tags.
    """
    for token in sentence:
        if token.is_punct or token.is_space:
            continue
        return _STARTER_CATEGORIES.get(token.pos_, "other")
    return "other"


def _compute_starter_entropy(distribution: dict[str, int], total_sentences: int) -> float:
    """Compute the normalized Shannon entropy of sentence starter categories.

    Entropy is normalized by the number of observed categories: 0.0 when all sentences start the same way, 1.0 for a
    perfectly uniform distribution.

    Args:
        distribution (dict[str, int]): Count of sentences per starter category.
        total_sentences (int): Total number of classified sentences.

    Returns:
        float: Normalized entropy in [0.0, 1.0], rounded to 4 decimal places.
    """
    counts = [count for count in distribution.values() if count > 0]
    if len(counts) <= 1:
        return 0.0
    entropy = -sum((count / total_sentences) * math.log2(count / total_sentences) for count in counts)
    return round(entropy / math.log2(len(counts)), 4)


def _compute_max_starter_run(categories: list[str]) -> int:
    """Compute the longest run of consecutive sentences sharing a starter category.

    Args:
        categories (list[str]): Starter categories of each sentence, in document order.

    Returns:
        int: Length of the longest consecutive run of identical starter categories.
    """
    max_run = 0
    current_run = 0
    previous_category = None
    for category in categories:
        current_run = current_run + 1 if category == previous_category else 1
        max_run = max(max_run, current_run)
        previous_category = category
    return max_run


def compute_rhythm_metrics(
    doc: Doc,
    short_threshold: int = DEFAULT_SHORT_THRESHOLD,
    long_threshold: int = DEFAULT_LONG_THRESHOLD,
) -> RhythmMetrics:
    """Compute sentence rhythm dispersion metrics, punctuation distribution and sentence starter variety.

    Args:
        doc (Doc): The parsed spaCy Doc instance.
        short_threshold (int): Upper word count bound for short sentences (< threshold). Defaults to 10 words.
        long_threshold (int): Lower word count bound for long sentences (> threshold). Defaults to 30 words.

    Returns:
        RhythmMetrics: An instance containing sentence length variations, punctuations counts and sentence starter
            variety.

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
        >>> metrics.starter_entropy
        1.0
        >>> metrics.max_consecutive_starter_run
        1
    """
    sentence_word_lengths: list[int] = []
    starter_categories: list[str] = []

    # Compute word count and starter category per active sentence
    for sent in doc.sents:
        word_count = sum(1 for tok in sent if not (tok.is_punct or tok.is_space))
        if word_count > 0:
            sentence_word_lengths.append(word_count)
            starter_categories.append(_classify_starter(sentence=sent))

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

    # Sentence starter variety
    raw_starter_counter: Counter[str] = Counter(starter_categories)
    starter_category_distribution: dict[str, int] = {
        category: raw_starter_counter.get(category, 0) for category in _STARTER_CATEGORY_KEYS
    }
    starter_entropy = _compute_starter_entropy(
        distribution=starter_category_distribution, total_sentences=total_sentences
    )
    pronoun_starter_ratio = (
        round(starter_category_distribution["pronoun"] / total_sentences, 4) if total_sentences > 0 else 0.0
    )
    max_consecutive_starter_run = _compute_max_starter_run(categories=starter_categories)

    return RhythmMetrics(
        avg_sentence_length=avg_sentence_length,
        sentence_length_variance=sentence_length_variance,
        sentence_length_std_dev=sentence_length_std_dev,
        short_sentence_ratio=short_sentence_ratio,
        long_sentence_ratio=long_sentence_ratio,
        punctuation_distribution=punctuation_distribution,
        starter_category_distribution=starter_category_distribution,
        starter_entropy=starter_entropy,
        pronoun_starter_ratio=pronoun_starter_ratio,
        max_consecutive_starter_run=max_consecutive_starter_run,
    )
