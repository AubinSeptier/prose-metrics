"""Data models representing the text analysis reports."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class VolumeMetrics:
    """Volumetric measurements of the text.

    Attributes:
        character_count (int): Total number of characters including spaces.
        character_count_no_spaces (int): Total number of characters excluding spaces.
        word_count (int): Total count of words (excluding punctuation tokens).
        sentence_count (int): Total number of detected sentences.
        paragraph_count (int): Total number of paragraphs.
        dialogue_word_count (int): Word count located inside detected dialogues.
        narrative_word_count (int): Word count located outside detected dialogues.
        dialogue_ratio (float): Proportion of dialogue words over total words (0.0 to 1.0).
    """

    character_count: int
    character_count_no_spaces: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    dialogue_word_count: int
    narrative_word_count: int
    dialogue_ratio: float


@dataclass(slots=True, frozen=True)
class RhythmMetrics:
    """Rhythm and sentence structure dispersion metrics.

    Dict field is excluded from hashing to keep reports hashable.

    Attributes:
        avg_sentence_length (float): Mean sentence length in words.
        sentence_length_variance (float): Variance of sentence lengths (cadence variation).
        sentence_length_std_dev (float): Standard deviation of sentence lengths.
        short_sentence_ratio (float): Proportion of short sentences (e.g. < 10 words).
        long_sentence_ratio (float): Proportion of long sentences (e.g. > 30 words).
        punctuation_distribution (dict[str, int]): Distribution of specific punctuation marks.
    """

    avg_sentence_length: float
    sentence_length_variance: float
    sentence_length_std_dev: float
    short_sentence_ratio: float
    long_sentence_ratio: float
    punctuation_distribution: dict[str, int] = field(hash=False)


@dataclass(slots=True, frozen=True)
class StyleMetrics:
    """Stylistic and grammatical distribution metrics.

    Dict field is excluded excluded from hashing to keep reports hashable.

    Attributes:
        noun_ratio (float): Proportion of nouns relative to total words.
        verb_ratio (float): Proportion of verbs relative to total words.
        adjective_ratio (float): Proportion of adjectives relative to total words.
        adverb_ratio (float): Proportion of adverbs relative to total words.
        pronoun_ratio (float): Proportion of pronouns relative to total words.
        adverbs_manner_count (int): Count of manner adverbs detected via syntactic and lexical rules.
        pos_distribution (dict[str, int]): Raw count of universal POS tags.
    """

    noun_ratio: float
    verb_ratio: float
    adjective_ratio: float
    adverb_ratio: float
    pronoun_ratio: float
    adverbs_manner_count: int
    pos_distribution: dict[str, int] = field(hash=False)


@dataclass(slots=True, frozen=True)
class VocabularyMetrics:
    """Lexical richness and vocabulary metrics.

    Attributes:
        unique_word_count (int): Number of distinct lemmas/types.
        ttr (float): Type-Token Ratio (unique words / total words).
        mattr (float): Moving Average Type-Token Ratio (sliding window).
        mattr_window_size (int): Window size used for MATTR calculation.
        hapax_count (int): Number of words appearing exactly once (hapax legomena).
        hapax_ratio (float): Proportion of hapax legomena relative to unique words.
    """

    unique_word_count: int
    ttr: float
    mattr: float
    mattr_window_size: int
    hapax_count: int
    hapax_ratio: float


@dataclass(slots=True, frozen=True)
class ReadabilityMetrics:
    """Readability and accessibility metrics.

    Attributes:
        flesch_reading_ease (float): Flesch Reading Ease score (higher is easier to read).
        flesch_kincaid_grade (float): Flesch-Kincaid Grade Level.
        gunning_fog (float): Gunning Fog Index.
        estimated_reading_time_minutes (float): Reading time estimate in minutes based on average reading speed.
    """

    flesch_reading_ease: float
    flesch_kincaid_grade: float
    gunning_fog: float
    estimated_reading_time_minutes: float


@dataclass(slots=True, frozen=True)
class RepetitionMetrics:
    """Close lexical repetition metrics.

    Repetitions are detected over content words only (nouns, adjectives, verbs, and adverbs) using lemmas by default.
    Distances are measured in content-word positions, not raw tokens.

    Attributes:
        repetition_density (float): Proportion of close repetitions over lexical words (0.0 to 1.0).
        close_repetition_count (int): Number of content-word occurrences repeating a word already seen within
            the window. Only the second and subsequent occurrences are counted.
        lexical_word_count (int): Total number of content words (nouns, adjectives, verbs, adverbs) considered
            for repetition detection. Denominator of repetition_density.
        window_size (int): Maximum distance, in content words, for two occurrences of the same word to be considered
            a close repetition.
    """

    repetition_density: float
    close_repetition_count: int
    lexical_word_count: int
    window_size: int


@dataclass(slots=True, frozen=True)
class DialogueMetrics:
    """Dialogue tag (parenthetical verb) metrics.

    Reporting verbs governing dialogue lines are detected via curated speech-verb lexicons combined
    with a proximity search around dialogue spans. Dash-style dialogues are not analyzed for tags,
    their spans capture the full line, incise included.

    Attributes:
        dialogue_verb_count (int): Total number of unique reporting verbs detected next to dialogue spans.
            Equal to neutral + expressive counts.
        neutral_dialogue_verb_count (int): Reporting verbs from the neutral lexicon (e.g., say, ask).
        expressive_dialogue_verb_count (int): Reporting verbs from the expressive (marked) lexicon (e.g., whisper,
            retort).
        neutral_dialogue_verb_ratio (float): Proportion of neutral reporting verbs over total reporting verbs
            (0.0 to 1.0).
    """

    dialogue_verb_count: int
    neutral_dialogue_verb_count: int
    expressive_dialogue_verb_count: int
    neutral_dialogue_verb_ratio: float


@dataclass(slots=True, frozen=True)
class TextReport:
    """Aggregated report containing all text metrics and analysis metadata.

    Attributes:
        language (str): Detected or specified language code (ISO 639-1, e.g., "en" for English).
        execution_time_seconds (float): Total processing time in seconds.
        volume (VolumeMetrics | None): Volumetric measurements of the text.
        rhythm (RhythmMetrics | None): Rhythm and sentence structure dispersion metrics.
        style (StyleMetrics | None): Stylistic and grammatical distribution metrics.
        vocabulary (VocabularyMetrics | None): Lexical richness and vocabulary metrics.
        readability (ReadabilityMetrics | None): Readability and accessibility metrics.
        repetition (RepetitionMetrics | None): Repetition and redundancy metrics.
        dialogue (DialogueMetrics | None): Dialogue-specific metrics.
    """

    language: str
    execution_time_seconds: float
    volume: VolumeMetrics | None = None
    rhythm: RhythmMetrics | None = None
    style: StyleMetrics | None = None
    vocabulary: VocabularyMetrics | None = None
    readability: ReadabilityMetrics | None = None
    repetition: RepetitionMetrics | None = None
    dialogue: DialogueMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the TextReport dataclass and all nested dataclasses to a dictionary.

        Returns:
            dict: A dictionary representation of the TextReport.

        Examples:
            >>> from prose_metrics import analyze
            >>> report = analyze("The cat sat on the mat.")
            >>> data = report.to_dict()
            >>> data["volume"]["word_count"]
            6
            >>> sorted(data.keys())
            ['dialogue', 'execution_time_seconds', 'language', 'readability', 'repetition', 'rhythm', 'style',
            'vocabulary', 'volume']
        """
        return asdict(self)
