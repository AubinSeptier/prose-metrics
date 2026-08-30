"""Close lexical repetition metrics calculation module."""

from typing import Final

from spacy.tokens import Doc

from prose_metrics.models.report import RepetitionMetrics

# Universal POS tags considered lexical content words for repetition detection, excluding proper nouns.
_CONTENT_WORD_POS: Final[frozenset[str]] = frozenset({"NOUN", "ADJ", "VERB", "ADV"})

DEFAULT_WINDOW_SIZE: Final[int] = 50


def _extract_lexical_words(doc: Doc, use_lemmas: bool) -> list[str]:
    """Extract normalized content-word representations from a parsed spaCy Doc.

    Only tokens whose universal POS tag denotes a content word (noun, adjective, verb, or adverb) are kept.
    The filter also excludes punctuation, stopwords, and proper nouns.

    Args:
        doc (Doc): The parsed spaCy Doc instance.
        use_lemmas (bool): If True, uses normalized lemmas. If False, uses raw lowercased tokens.

    Returns:
        list[str]: The normalized content words, in document order.
    """
    words: list[str] = []
    for token in doc:
        if token.pos_ not in _CONTENT_WORD_POS:
            continue
        words.append(token.lemma_.lower() if use_lemmas else token.text.lower())
    return words


def compute_repetition_metrics(
    doc: Doc,
    window_size: int = DEFAULT_WINDOW_SIZE,
    use_lemmas: bool = True,
) -> RepetitionMetrics:
    """Compute close lexical repetition metrics from a parsed spaCy Doc.

    Detects content words (nouns, adjectives, verbs, adverbs) whose previous occurrence of the same word lies at most
    window_size content words back. Only the second and subsequent occurrences are counted, in a single O(N) pass.

    Args:
        doc (Doc): The parsed spaCy Doc instance.
        window_size (int): Maximum distance, in content words, for two occurrences of the same word to be considered
            a close repetition. Defaults to 50.
        use_lemmas (bool): If True, uses normalized lemmas. If False, uses raw lowercased tokens. Defaults to True.

    Returns:
        RepetitionMetrics: An instance containing the repetition density, the close repetition count, the lexical word
            count, and the window size used for detection.

    Raises:
        ValueError: If window_size is less than 1.

    Examples:
        >>> from prose_metrics.nlp.pipeline import SpacyPipelineManager
        >>> nlp = SpacyPipelineManager().get_pipeline("en")
        >>> doc = nlp("The dog barked loudly. The cat ran fast. The dog slept peacefully.")
        >>> metrics = compute_repetition_metrics(doc)
        >>> metrics.lexical_word_count
        9
        >>> metrics.close_repetition_count
        1
        >>> metrics.repetition_density
        0.1111
    """
    if window_size < 1:
        msg = f"window_size must be >= 1, got {window_size}"
        raise ValueError(msg)

    words = _extract_lexical_words(doc=doc, use_lemmas=use_lemmas)
    lexical_word_count = len(words)

    last_seen: dict[str, int] = {}
    close_repetition_count = 0
    for i, word in enumerate(words):
        previous = last_seen.get(word)
        if previous is not None and i - previous <= window_size:
            close_repetition_count += 1
        last_seen[word] = i

    repetition_density = round(close_repetition_count / lexical_word_count, 4) if lexical_word_count else 0.0

    return RepetitionMetrics(
        repetition_density=repetition_density,
        close_repetition_count=close_repetition_count,
        lexical_word_count=lexical_word_count,
        window_size=window_size,
    )
