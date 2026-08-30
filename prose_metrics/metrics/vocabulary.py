"""Vocabulary and lexical richness metrics calculation module."""

import math
from collections import Counter
from typing import Final

from spacy.tokens import Doc

from prose_metrics.models.report import VocabularyMetrics

DEFAULT_MATTR_WINDOW_SIZE: Final[int] = 100
DEFAULT_MSTTR_SEGMENT_SIZE: Final[int] = 100


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


def _compute_yule_k(token_counter: Counter[str], total_words: int) -> float:
    """Compute Yule's K index from a word frequency counter.

    K = 10^4 * (sum of i^2 * V(i) - N) / N^2, where V(i) is the number of types occurring exactly i times and N is the
    total number of tokens. The index is largely independent of text length. Unlike TTR-based measures, a higher K
    means more repetition, i.e., lower lexical diversity.

    Args:
        token_counter (Counter[str]): Frequency of each normalized word or lemma.
        total_words (int): Total number of tokens.

    Returns:
        float: Yule's K, rounded to 4 decimal places.
    """
    spectrum: Counter[int] = Counter(token_counter.values())
    sum_i2_vi = sum(i * i * v_i for i, v_i in spectrum.items())
    return round(10**4 * (sum_i2_vi - total_words) / total_words**2, 4)


def _compute_maas_a2(total_words: int, unique_word_count: int) -> float:
    """Compute Maas's a^2 index of lexical diversity.

    a^2 = (ln N - ln V) / (ln N)^2, where N is the total number of tokens and V the number of distinct types.
    Unlike most measures, a lower a^2 means richer vocabulary. Returns 0.0 for texts shorter than 2 tokens,
    where the index is undefined (ln 1 = 0).

    Args:
        total_words (int): Total number of tokens.
        unique_word_count (int): Number of distinct types.

    Returns:
        float: Maas's a^2, rounded to 4 decimal places.
    """
    if total_words < 2:
        return 0.0
    log_n = math.log(total_words)
    return round((log_n - math.log(unique_word_count)) / log_n**2, 4)


def _compute_msttr(tokens: list[str], segment_size: int) -> float:
    """Compute Mean Segmental Type-Token Ratio (MSTTR) over disjoint segments.

    The text is split into consecutive non-overlapping segments of segment_size tokens, the TTR of each full segment is
    averaged. The trailing partial segment is discarded, so every segment contributes equally. If no full segment fits
    in the text, MSTTR falls back to standard TTR.

    Args:
        tokens (list[str]): Sequence of normalized words or lemmas.
        segment_size (int): Length of each disjoint segment.

    Returns:
        float: The MSTTR value, rounded to 4 decimal places.
    """
    total_tokens = len(tokens)
    if total_tokens == 0:
        return 0.0

    num_full_segments = total_tokens // segment_size
    if num_full_segments == 0:
        return round(len(set(tokens)) / total_tokens, 4)

    ttr_sum = 0.0
    for i in range(num_full_segments):
        segment = tokens[i * segment_size : (i + 1) * segment_size]
        ttr_sum += len(set(segment)) / segment_size

    return round(ttr_sum / num_full_segments, 4)


def compute_vocabulary_metrics(
    doc: Doc,
    mattr_window_size: int = DEFAULT_MATTR_WINDOW_SIZE,
    msttr_segment_size: int = DEFAULT_MSTTR_SEGMENT_SIZE,
    use_lemmas: bool = True,
) -> VocabularyMetrics:
    """Compute vocabulary richness, MATTR, and hapax legomena metrics.

    Args:
        doc (Doc): The parsed spaCy Doc instance.
        mattr_window_size (int): Window size for MATTR calculation. Default is 100 tokens.
        msttr_segment_size (int): Segment size for MSTTR calculation. Default is 100 tokens.
        use_lemmas (bool): If True, uses normalized lemmas. If False, uses raw lower tokens.

    Returns:
        VocabularyMetrics: An instance containing all computed vocabulary metrics (unique word count,
            TTR, MATTR, hapax count, hapax ratio, Yule's K, Maas's a^2, and MSTTR).

    Raises:
        ValueError: If mattr_window_size or msttr_segment_size is less than 1.

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
        >>> metrics.yule_k
        1666.6667
        >>> metrics.maas_a2
        0.1263
        >>> metrics.msttr
        0.6667
    """
    if mattr_window_size < 1:
        msg = f"mattr_window_size must be >= 1, got {mattr_window_size}"
        raise ValueError(msg)
    if msttr_segment_size < 1:
        msg = f"msttr_segment_size must be >= 1, got {msttr_segment_size}"
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
            yule_k=0.0,
            maas_a2=0.0,
            msttr=0.0,
            msttr_segment_size=msttr_segment_size,
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

    # Yule's K
    yule_k = _compute_yule_k(token_counter=token_counter, total_words=total_words)

    # Maas's a^2
    maas_a2 = _compute_maas_a2(total_words=total_words, unique_word_count=unique_word_count)

    # Mean Segmental Type-Token Ratio (MSTTR)
    msttr = _compute_msttr(tokens=tokens, segment_size=msttr_segment_size)

    return VocabularyMetrics(
        unique_word_count=unique_word_count,
        ttr=ttr,
        mattr=mattr,
        mattr_window_size=mattr_window_size,
        hapax_count=hapax_count,
        hapax_ratio=hapax_ratio,
        yule_k=yule_k,
        maas_a2=maas_a2,
        msttr=msttr,
        msttr_segment_size=msttr_segment_size,
    )
