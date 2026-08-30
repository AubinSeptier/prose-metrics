"""Dialogue tag (parenthetical verb) metrics calculation module."""

import bisect
import re
from typing import Final

from spacy.tokens import Doc, Token

from prose_metrics.models.report import DialogueMetrics

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

# Maximum number of content tokens between a dialogue span boundary and its reporting verb
_MAX_TAG_DISTANCE: Final[int] = 8

# Speech-verb lexicons keyed by language, split into neutral and expressive verbs
# Lemmas are used, so inflected and reflexive forms are covered (e.g., "ask", "asks", "asked")
_NEUTRAL_SPEECH_VERBS: Final[dict[str, frozenset[str]]] = {
    "en": frozenset({"say", "ask", "answer", "reply", "add", "continue", "tell"}),
    "fr": frozenset({"dire", "demander", "répondre", "ajouter", "continuer", "raconter"}),
}

_EXPRESSIVE_SPEECH_VERBS: Final[dict[str, frozenset[str]]] = {
    "en": frozenset(
        {
            "whisper",
            "murmur",
            "mutter",
            "shout",
            "yell",
            "scream",
            "exclaim",
            "retort",
            "snap",
            "thunder",
            "hiss",
            "growl",
            "sigh",
            "stammer",
            "cry",
            "bellow",
            "roar",
        }
    ),
    "fr": frozenset(
        {
            "murmurer",
            "chuchoter",
            "crier",
            "hurler",
            "écrier",
            "exclamer",
            "rétorquer",
            "répliquer",
            "tonner",
            "grogner",
            "gémir",
            "marmonner",
            "bredouiller",
            "souffler",
            "sangloter",
            "rugir",
        }
    ),
}


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


def _find_tag_verb(
    tokens: list[Token],
    index: int,
    step: int,
    spans: list[tuple[int, int]],
    span_starts: list[int],
    lexicon: frozenset[str],
) -> Token | None:
    """Walk tokens in one direction from a dialogue span boundary to find a reporting verb.

    The search is bounded by three guards: the sentence of the first content token encountered (crossing into another
    sentence stops the walk), other dialogue spans, and a maximum distance of _MAX_TAG_DISTANCE content tokens.
    Taking the first content token's sentence as reference (rather than the closing delimiter's) handles spaCy splitting
    the sentence after dialogue-final "!", or "?", when the tag starts its own sentence.

    Args:
        tokens (list[Token]): All tokens of the Doc, in order.
        index (int): Index of the first token to inspect (right or left of the dialogue span).
        step (int): Walk direction: 1 for right side, -1 for left side.
        spans (list[tuple[int, int]]): Sorted, non-overlapping dialogue spans.
        span_starts (list[int]): Pre-extracted span start offsets for binary search.
        lexicon (frozenset[str]): Combined neutral and expressive speech-verb lemmas.

    Returns:
        Token | None: The first VERB token whose lemma is in the lexicon, or None.
    """
    reference_sentence = None
    distance = 0
    i = index

    while 0 <= i < len(tokens):
        token = tokens[i]
        i += step
        if token.is_space or token.is_punct:
            continue
        if reference_sentence is None:
            reference_sentence = token.sent
        elif token.sent != reference_sentence:
            break
        if _is_token_in_spans(token_idx=token.idx, sorted_spans=spans, span_starts=span_starts):
            break
        distance += 1
        if distance > _MAX_TAG_DISTANCE:
            break
        if token.pos_ == "VERB" and token.lemma_.lower() in lexicon:
            return token
    return None


def compute_dialogue_metrics(text: str, doc: Doc) -> DialogueMetrics:
    """Compute dialogue tag (parenthetical verb) metrics from raw text and its parsed spaCy Doc.

    For each dialogue span, a reporting verb is searched on the right side first (incise position, e.g.
    "she whispered"), then on the left side (pre-dialogue position, e.g., 'John said: "..."').
    Only verbs from curated speech-verb lexicons count. Verbs are deduplicated by token index, so a tag
    splitting two dialogue spans is counted once.

    Known limitations: dash-style dialogues capture the incise inside the span, so their tags are not detected;
    narrative uses of speech verbs next to dialogue (e.g., "He said nothing") may be counted as tags.

    Args:
        text (str): The raw input string.
        doc (Doc): The parsed spaCy Doc instance.

    Returns:
        DialogueMetrics: An instance containing total, neutral, and expressive reporting verb counts, and the neutral
            verb ratio.

    Examples:
        >>> from prose_metrics.nlp.pipeline import SpacyPipelineManager
        >>> nlp = SpacyPipelineManager().get_pipeline("en")
        >>> text = "“Run now,” she whispered. “Wait,” he said."
        >>> metrics = compute_dialogue_metrics(text, nlp(text))
        >>> metrics.dialogue_verb_count
        2
        >>> metrics.neutral_dialogue_verb_count
        1
        >>> metrics.expressive_dialogue_verb_count
        1
        >>> metrics.neutral_dialogue_verb_ratio
        0.5
    """
    language = doc.lang_
    neutral_lexicon = _NEUTRAL_SPEECH_VERBS.get(language, frozenset())
    expressive_lexicon = _EXPRESSIVE_SPEECH_VERBS.get(language, frozenset())
    lexicon = neutral_lexicon | expressive_lexicon

    zeroed = DialogueMetrics(
        dialogue_verb_count=0,
        neutral_dialogue_verb_count=0,
        expressive_dialogue_verb_count=0,
        neutral_dialogue_verb_ratio=0.0,
    )
    if not lexicon:
        return zeroed

    spans = _extract_dialogue_spans(text=text)
    if not spans:
        return zeroed

    span_starts = [span[0] for span in spans]
    tokens = list(doc)
    offsets = [token.idx for token in tokens]

    seen: set[int] = set()
    neutral_count = 0
    expressive_count = 0
    for start, end in spans:
        tag = _find_tag_verb(
            tokens=tokens,
            index=bisect.bisect_left(offsets, end),
            step=1,
            spans=spans,
            span_starts=span_starts,
            lexicon=lexicon,
        )
        if tag is None:
            tag = _find_tag_verb(
                tokens=tokens,
                index=bisect.bisect_left(offsets, start) - 1,
                step=-1,
                spans=spans,
                span_starts=span_starts,
                lexicon=lexicon,
            )
        if tag is None or tag.i in seen:
            continue
        seen.add(tag.i)
        if tag.lemma_.lower() in neutral_lexicon:
            neutral_count += 1
        else:
            expressive_count += 1

    total = neutral_count + expressive_count
    return DialogueMetrics(
        dialogue_verb_count=total,
        neutral_dialogue_verb_count=neutral_count,
        expressive_dialogue_verb_count=expressive_count,
        neutral_dialogue_verb_ratio=round(neutral_count / total, 4) if total else 0.0,
    )
