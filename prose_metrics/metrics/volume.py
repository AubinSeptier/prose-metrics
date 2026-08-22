"""Volumetric metrics calculation module for text."""

import bisect
import re

from spacy.tokens import Doc

from prose_metrics.models.report import VolumeMetrics

# Regex patterns for various dialogue conventions in fictional texts
_DIALOGUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # English curly double quotes: “...”
    re.compile(r"“(?P<dialogue>[^”]+)”"),
    # French guillemets: « ... »
    re.compile(r"«(?P<dialogue>[^»]+)»"),
    # Straight double quotes: "..."
    re.compile(r'"(?P<dialogue>[^"\n]+)"'),
    # Dialogue dashes at line start: —, –, or -
    re.compile(r"^[—–\-]\s*(?P<dialogue>.+)$", flags=re.MULTILINE),
)


def _extract_dialogue_spans(text: str) -> list[tuple[int, int]]:
    """Extract and merge all character index spans corresponding to dialogue.

    Args:
        text (str): The raw input string.

    Returns:
        list[tuple[int, int]]: A sorted list of non-overlapping (start, end) character tuples.
    """
    raw_spans: list[tuple[int, int]] = []

    for pattern in _DIALOGUE_PATTERNS:
        for match in pattern.finditer(text):
            # Capture only the dialogue content (excluding delimiters)
            start, end = match.span("dialogue")
            raw_spans.append((start, end))

    if not raw_spans:
        return []

    # Sort spans by start index
    raw_spans.sort(key=lambda span: span[0])

    # Merge overlapping or contiguous spans
    merged_spans: list[tuple[int, int]] = [raw_spans[0]]
    for current_start, current_end in raw_spans[1:]:
        prev_start, prev_end = merged_spans[-1]
        if current_start <= prev_end:
            merged_spans[-1] = (prev_start, max(prev_end, current_end))
        else:
            merged_spans.append((current_start, current_end))

    return merged_spans


def _is_token_in_spans(token_idx: int, sorted_spans: list[tuple[int, int]], span_starts: list[int]) -> bool:
    """Check if a token index falls within any dialogue span using binary search.

    Args:
        token_idx (int): The starting character index of the token.
        sorted_spans (list[tuple[int, int]]): The list of non-overlapping (start, end) spans.
        span_starts (list[int]): The pre-extracted list of span start indices.

    Returns:
        bool: True if the token is inside a dialogue span, False otherwise.
    """
    if not sorted_spans:
        return False

    # Find the rightmost span whose start is <= token_idx
    idx = bisect.bisect_right(span_starts, token_idx) - 1
    if idx >= 0:
        start, end = sorted_spans[idx]
        if start <= token_idx < end:
            return True

    return False


def compute_volume_metrics(text: str, doc: Doc) -> VolumeMetrics:
    """Compute all volumetric metrics from raw text and its parsed spaCy Doc.

    Args:
        text (str): The raw text string.
        doc (Doc): The parsed spaCy Doc instance.

    Returns:
        VolumeMetrics: An instance containing all computed volumetric metrics (character, word,
            sentence, paragraph, dialogue word and narrative word count, and dialogue ratio).
    """
    # Characters
    character_count = len(text)
    character_count_no_spaces = sum(1 for char in text if not char.isspace())

    # Paragraphs (non-empty lines separated by newlines)
    paragraphs = [p for p in text.splitlines() if p.strip()]
    paragraph_count = len(paragraphs)

    # Sentences (from spaCy senter/parser)
    # Filter out sentences that contain only spaces or punctuation
    sentences = [sent for sent in doc.sents if any(not (tok.is_punct or tok.is_space) for tok in sent)]
    sentence_count = len(sentences)

    # Words and Dialogues
    dialogue_spans = _extract_dialogue_spans(text)
    span_starts = [span[0] for span in dialogue_spans]

    word_count = 0
    dialogue_word_count = 0
    for token in doc:
        # Ignore punctuation and standalone whitespace tokens
        if token.is_punct or token.is_space:
            continue

        word_count += 1
        if _is_token_in_spans(token_idx=token.idx, sorted_spans=dialogue_spans, span_starts=span_starts):
            dialogue_word_count += 1

    narrative_word_count = word_count - dialogue_word_count
    dialogue_ratio = round(dialogue_word_count / word_count, 4) if word_count > 0 else 0.0

    return VolumeMetrics(
        character_count=character_count,
        character_count_no_spaces=character_count_no_spaces,
        word_count=word_count,
        sentence_count=sentence_count,
        paragraph_count=paragraph_count,
        dialogue_word_count=dialogue_word_count,
        narrative_word_count=narrative_word_count,
        dialogue_ratio=dialogue_ratio,
    )
