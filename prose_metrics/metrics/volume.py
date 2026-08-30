"""Volumetric metrics calculation module for text."""

from spacy.tokens import Doc

from prose_metrics.metrics.dialogue import _extract_dialogue_spans, _is_token_in_spans
from prose_metrics.models.report import VolumeMetrics


def compute_volume_metrics(text: str, doc: Doc) -> VolumeMetrics:
    r"""Compute all volumetric metrics from raw text and its parsed spaCy Doc.

    Args:
        text (str): The raw text string.
        doc (Doc): The parsed spaCy Doc instance.

    Returns:
        VolumeMetrics: An instance containing all computed volumetric metrics (character, word,
            sentence, paragraph, dialogue word and narrative word count, and dialogue ratio).

    Examples:
        >>> from prose_metrics.nlp.pipeline import SpacyPipelineManager
        >>> nlp = SpacyPipelineManager().get_pipeline("en")
        >>> text = "She whispered, “Run now.”\nHe stayed."
        >>> metrics = compute_volume_metrics(text, nlp(text))
        >>> metrics.word_count
        6
        >>> metrics.sentence_count
        2
        >>> metrics.dialogue_word_count
        2
        >>> metrics.dialogue_ratio
        0.3333
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
