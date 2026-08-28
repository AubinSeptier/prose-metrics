"""Vocabulary and lexical richness metrics calculation module."""

from collections import Counter
from typing import Final

from spacy.tokens import Doc

from prose_metrics.models.report import VocabularyMetrics

DEFAULT_MATTR_WINDOW_SIZE: Final[int] = 100


def _compute_mattr(tokens: list[str], window_size: int) -> float:
    """Compute Moving Average Type-Token Ratio (MATTR) using an O(N) sliding window.

    Args:
        tokens (list[str]): Sequence of normalized words or lemmas.
        window_size (int): Length of the sliding window.

    Returns:
        float: The MATTR value rounded to 4 decimal places.
    """
    total_tokens = len(tokens)
    if total_tokens == 0:
        return 0.0

    # If text is shorter than window, MATTR falls back to standard TTR
    if total_tokens <= window_size:
        return round(len(set(tokens)) / total_tokens, 4)

    # Initialize first window [0 : window_size]
    window_counts: dict[str, int] = {}
    for i in range(window_size):
        token = tokens[i]
        window_counts[token] = window_counts.get(token, 0) + 1

    unique_types = len(window_counts)
    total_unique_types = unique_types

    # Slide window token by token across the text
    num_windows = total_tokens - window_size + 1
    for i in range(1, num_windows):
        token_out = tokens[i - 1]
        token_in = tokens[i + window_size - 1]

        # Remove outgoing token
        if window_counts[token_out] == 1:
            del window_counts[token_out]
            unique_types -= 1
        else:
            window_counts[token_out] -= 1

        # Insert incoming token
        if token_in not in window_counts:
            window_counts[token_in] = 1
            unique_types += 1
        else:
            window_counts[token_in] += 1

        total_unique_types += unique_types

    # Mean TTR over all sliding windows
    mean_unique = total_unique_types / num_windows
    return round(mean_unique / window_size, 4)


def compute_vocabulary_metrics(
    doc: Doc,
    mattr_window_size: int = DEFAULT_MATTR_WINDOW_SIZE,
    use_lemmas: bool = True,
) -> VocabularyMetrics:
    """Compute vocabulary richness, MATTR, and hapax legomena metrics.

    Args:
        doc (Doc): The parsed spaCy Doc instance.
        mattr_window_size (int): Window size for MATTR calculation. Default is 100 tokens.
        use_lemmas (bool): If True, uses normalized lemmas. If False, uses raw lower tokens.

    Returns:
        VocabularyMetrics: An instance containing all computed vocabulary metrics (unique word count,
            TTR, MATTR, hapax count, and hapax ratio).

    Raises:
        ValueError: If mattr_window_size is less than 1.

    Examples:
        >>> from prose_metrics.nlp.pipeline import SpacyPipelineManager
        >>> nlp = SpacyPipelineManager().get_pipeline("en")
        >>> doc = nlp("I run, he runs, she ran.")
        >>> metrics = compute_vocabulary_metrics(doc)
        >>> metrics.unique_word_count
        4
        >>> metrics.ttr
        0.6667
        >>> metrics.mattr
        0.6667
        >>> metrics.hapax_count
        3
        >>> compute_vocabulary_metrics(doc, use_lemmas=False).ttr
        1.0
    """
    if mattr_window_size < 1:
        msg = f"mattr_window_size must be >= 1, got {mattr_window_size}"
        raise ValueError(msg)

    tokens: list[str] = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue

        token_repr = token.lemma_.lower() if use_lemmas else token.text.lower()
        tokens.append(token_repr)

    total_words = len(tokens)

    if total_words == 0:
        return VocabularyMetrics(
            unique_word_count=0,
            ttr=0.0,
            mattr=0.0,
            mattr_window_size=mattr_window_size,
            hapax_count=0,
            hapax_ratio=0.0,
        )

    # Frequency analysis
    token_counter = Counter(tokens)
    unique_word_count = len(token_counter)

    # Type-Token Ratio (TTR)
    ttr = round(unique_word_count / total_words, 4)

    # Hapax Legomena - lemmas occurring exactly once
    hapax_count = sum(1 for count in token_counter.values() if count == 1)
    hapax_ratio = round(hapax_count / unique_word_count, 4) if unique_word_count > 0 else 0.0

    # Moving Average Type-Token Ratio (MATTR)
    mattr = _compute_mattr(tokens, mattr_window_size)

    return VocabularyMetrics(
        unique_word_count=unique_word_count,
        ttr=ttr,
        mattr=mattr,
        mattr_window_size=mattr_window_size,
        hapax_count=hapax_count,
        hapax_ratio=hapax_ratio,
    )
