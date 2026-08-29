"""Specific dialogue metrics calculation module."""

import bisect
import re

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
