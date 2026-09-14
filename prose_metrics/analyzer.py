"""Central orchestrator for prose and text analysis."""

import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace
from typing import Final, Literal

from spacy.tokens import Doc

from prose_metrics.metrics import (
    compute_dialogue_metrics,
    compute_readability_metrics,
    compute_repetition_metrics,
    compute_rhythm_metrics,
    compute_style_metrics,
    compute_vocabulary_metrics,
    compute_volume_metrics,
)
from prose_metrics.models.report import (
    DialogueMetrics,
    ReadabilityMetrics,
    RepetitionMetrics,
    RhythmMetrics,
    StyleMetrics,
    TextReport,
    VocabularyMetrics,
    VolumeMetrics,
)
from prose_metrics.nlp.pipeline import SpacyPipelineManager

MetricName = Literal["dialogue", "readability", "repetition", "rhythm", "style", "vocabulary", "volume"]

AVAILABLE_METRICS: Final[frozenset[MetricName]] = frozenset(
    {"dialogue", "readability", "repetition", "rhythm", "style", "vocabulary", "volume"}
)


class TextAnalyzer:
    """Main analyzer coordinating NLP parsing and metric computations.

    Examples:
        >>> analyzer = TextAnalyzer(language="en")
        >>> report = analyzer.analyze("The cat sat on the mat.")
        >>> report.volume.word_count
        6
        >>> report.language
        'en'
    """

    def __init__(
        self,
        language: str = "en",
        model_name: str | None = None,
    ) -> None:
        """Initialize the analyzer with a specific language or model.

        Args:
            language (str): Target language code (ISO 639-1, e.g., "en" for English). Default is "en".
            model_name (str | None): Optional explicit spaCy model name.
        """
        self.language = language.lower()
        self.model_name = model_name
        self._pipeline_manager = SpacyPipelineManager()

    def analyze(
        self,
        text: str,
        doc: Doc | None = None,
        metrics: Sequence[MetricName] | Literal["all"] = "all",
        mattr_window_size: int = 100,
        msttr_segment_size: int = 100,
        words_per_minute: int = 200,
        short_threshold: int = 10,
        long_threshold: int = 30,
        use_lemmas: bool = True,
        repetition_window_size: int = 50,
    ) -> TextReport:
        """Analyze a given text and generate a structured TextReport.

        `execution_time_seconds`'s value covers parsing + metric computation, excluding argument validation.

        Args:
            text (str): The raw text string to analyze.
            doc (Doc | None): Optional pre-parsed spaCy Doc object to bypass re-tokenization. If None, the text will be
                parsed using the spaCy pipeline.
            metrics (Sequence[MetricName] | Literal["all"]): Metrics to calculate. Either 'all' or a sequence of metric
                names ("dialogue", "readability", "repetition", "rhythm", "style", "vocabulary", "volume"). If an empty
                list is provided, all metrics will be computed.
            mattr_window_size (int): Window size for MATTR calculation. Defaults to 100.
            msttr_segment_size (int): Segment size for MSTTR calculation. Defaults to 100.
            words_per_minute (int): Reading speed for reading time estimation. Defaults to 200.
            short_threshold (int): Upper word count bound for short sentences (< threshold). Defaults to 10 words.
            long_threshold (int): Lower word count bound for long sentences (> threshold). Defaults to 30 words.
            use_lemmas (bool): If True, uses normalized lemmas. If False, uses raw lower tokens. Defaults to True.
            repetition_window_size (int): Maximum distance, in content words, for two occurrences of the same word to be
                considered a close repetition. Defaults to 50.

        Returns:
            TextReport: A dataclass containing all computed metrics and analysis metadata.

        Raises:
            ValueError: If the metrics argument is invalid, i.e., not a valid metric name or sequence of metric names.
                Or if mattr_window_size, msttr_segment_size, repetition_window_size, or words_per_minute is less than 1.

        Examples:
            >>> analyzer = TextAnalyzer(language="en")
            >>> report = analyzer.analyze("The cat sat on the mat.", metrics=["volume", "readability"])
            >>> report.style is None
            True
            >>> analyzer.analyze("The cat sat on the mat.", metrics=["unknown"])
            Traceback (most recent call last):
                ...
            ValueError: Invalid metric(s): ['unknown']. Available metrics: ['dialogue', 'readability', 'repetition',
            'rhythm', 'style', 'vocabulary', 'volume']
        """
        # Resolve metrics to compute
        selected_metrics = _resolve_metrics(metrics=metrics)

        start_time = time.perf_counter()
        # Parse text with spaCy (if not already provided)
        if doc is None:
            nlp = self._pipeline_manager.get_pipeline(language=self.language, model_name=self.model_name)
            parsed_doc = nlp(text=text)
        else:
            parsed_doc = doc

        # Compute selected metrics
        report = self._compute_report(
            text=text,
            parsed_doc=parsed_doc,
            selected_metrics=selected_metrics,
            execution_time_seconds=0.0,
            mattr_window_size=mattr_window_size,
            msttr_segment_size=msttr_segment_size,
            words_per_minute=words_per_minute,
            short_threshold=short_threshold,
            long_threshold=long_threshold,
            use_lemmas=use_lemmas,
            repetition_window_size=repetition_window_size,
        )

        execution_time = round(time.perf_counter() - start_time, 4)
        return replace(report, execution_time_seconds=execution_time)

    def pipe(
        self,
        texts: Iterable[str],
        metrics: Sequence[MetricName] | Literal["all"] = "all",
        batch_size: int = 64,
        n_process: int = 1,
        mattr_window_size: int = 100,
        msttr_segment_size: int = 100,
        words_per_minute: int = 200,
        short_threshold: int = 10,
        long_threshold: int = 30,
        use_lemmas: bool = True,
        repetition_window_size: int = 50,
    ) -> Iterator[TextReport]:
        """Analyze multiple texts lazily using spaCy's batched pipe processing.

        Texts are parsed through `nlp.pipe()`, which batches the neural pipeline components and optionally distributes
        parsing across multiple processes. Reports are yielded one at a time, in input order, as parsing progresses.
        Texts and docs are paired via spaCy 'as_tuples=True'.

        Since parsing is batched, per-report parsing time cannot be isolated: `execution_time_seconds` is stamped with
        the running amortized time (cumulative elapsed time divided by the number of reports yielded so far).
        `execution_time_seconds`'s value covers parsing + metric computation, excluding argument validation.

        Args:
            texts (Iterable[str]): The raw text strings to analyze. May be a generator; it is consumed lazily.
            metrics (Sequence[MetricName] | Literal["all"]): Metrics to calculate. Either 'all' or a sequence of metric
                names ("dialogue", "readability", "repetition", "rhythm", "style", "vocabulary", "volume"). If an empty
                list is provided, all metrics will be computed.
            batch_size (int): Number of texts buffered per spaCy pipe batch. Defaults to 64.
            n_process (int): Number of processes for spaCy to use during parsing. Defaults to 1.
            mattr_window_size (int): Window size for MATTR calculation. Defaults to 100.
            msttr_segment_size (int): Segment size for MSTTR calculation. Defaults to 100.
            words_per_minute (int): Reading speed for reading time estimation. Defaults to 200.
            short_threshold (int): Upper word count bound for short sentences (< threshold). Defaults to 10 words.
            long_threshold (int): Lower word count bound for long sentences (> threshold). Defaults to 30 words.
            use_lemmas (bool): If True, uses normalized lemmas. If False, uses raw lower tokens. Defaults to True.
            repetition_window_size (int): Maximum distance, in content words, for two occurrences of the same word to be
                considered a close repetition. Defaults to 50.

        Returns:
            Iterator[TextReport]: A lazy iterator of reports, in the same order as the input texts.

        Raises:
            ValueError: if the metrics argument is invalid, or if batch_size or n_process is less than 1.

        Examples:
            >>> analyzer = TextAnalyzer(language="en")
            >>> reports = list(analyzer.pipe(["The cat sat.", "A dog runs fast."]))
            >>> [r.volume.word_count for r in reports]
            [4, 4]
        """
        if batch_size < 1:
            msg = f"batch_size must be at least 1, got {batch_size}"
            raise ValueError(msg)
        if n_process < 1:
            msg = f"n_process must be at least 1, got {n_process}"
            raise ValueError(msg)

        selected_metrics = _resolve_metrics(metrics=metrics)

        def _generate() -> Iterator[TextReport]:
            start_time = time.perf_counter()
            nlp = self._pipeline_manager.get_pipeline(language=self.language, model_name=self.model_name)
            text_pairs = ((text, text) for text in texts)
            docs = nlp.pipe(text_pairs, batch_size=batch_size, n_process=n_process, as_tuples=True)
            for count, (parsed_doc, text) in enumerate(docs, start=1):
                amortized_time = round((time.perf_counter() - start_time) / count, 4)
                yield self._compute_report(
                    text=text,
                    parsed_doc=parsed_doc,
                    selected_metrics=selected_metrics,
                    execution_time_seconds=amortized_time,
                    mattr_window_size=mattr_window_size,
                    msttr_segment_size=msttr_segment_size,
                    words_per_minute=words_per_minute,
                    short_threshold=short_threshold,
                    long_threshold=long_threshold,
                    use_lemmas=use_lemmas,
                    repetition_window_size=repetition_window_size,
                )

        return _generate()

    def _compute_report(
        self,
        text: str,
        parsed_doc: Doc,
        selected_metrics: frozenset[MetricName],
        execution_time_seconds: float,
        mattr_window_size: int,
        msttr_segment_size: int,
        words_per_minute: int,
        short_threshold: int,
        long_threshold: int,
        use_lemmas: bool,
        repetition_window_size: int,
    ) -> TextReport:
        """Compute the requested metrics and return a TextReport.

        Args:
            text (str): The raw text string to analyze.
            parsed_doc (Doc): The pre-parsed spaCy Doc object.
            selected_metrics (frozenset[MetricName]): Metrics to calculate.
            execution_time_seconds (float): Total processing time in seconds.
            mattr_window_size (int): Window size for MATTR calculation.
            msttr_segment_size (int): Segment size for MSTTR calculation.
            words_per_minute (int): Reading speed for reading time estimation.
            short_threshold (int): Upper word count bound for short sentences (< threshold).
            long_threshold (int): Lower word count bound for long sentences (> threshold).
            use_lemmas (bool): If True, uses normalized lemmas. If False, uses raw lower tokens.
            repetition_window_size (int): Maximum distance, in content words, for two occurrences of the same word to be
                considered a close repetition.

        Returns:
            TextReport: A dataclass containing all computed metrics and analysis metadata.
        """
        volume_metrics: VolumeMetrics | None = None
        rhythm_metrics: RhythmMetrics | None = None
        style_metrics: StyleMetrics | None = None
        vocabulary_metrics: VocabularyMetrics | None = None
        readability_metrics: ReadabilityMetrics | None = None
        repetition_metrics: RepetitionMetrics | None = None
        dialogue_metrics: DialogueMetrics | None = None

        if "volume" in selected_metrics:
            volume_metrics = compute_volume_metrics(text=text, doc=parsed_doc)

        if "rhythm" in selected_metrics:
            rhythm_metrics = compute_rhythm_metrics(
                doc=parsed_doc,
                short_threshold=short_threshold,
                long_threshold=long_threshold,
            )

        if "style" in selected_metrics:
            style_metrics = compute_style_metrics(doc=parsed_doc)

        if "vocabulary" in selected_metrics:
            vocabulary_metrics = compute_vocabulary_metrics(
                doc=parsed_doc,
                mattr_window_size=mattr_window_size,
                msttr_segment_size=msttr_segment_size,
                use_lemmas=use_lemmas,
            )

        if "readability" in selected_metrics:
            readability_metrics = compute_readability_metrics(
                text=text,
                doc=parsed_doc,
                language=self.language,
                words_per_minute=words_per_minute,
            )

        if "repetition" in selected_metrics:
            repetition_metrics = compute_repetition_metrics(
                doc=parsed_doc,
                window_size=repetition_window_size,
                use_lemmas=use_lemmas,
            )

        if "dialogue" in selected_metrics:
            dialogue_metrics = compute_dialogue_metrics(text=text, doc=parsed_doc)

        return TextReport(
            language=self.language,
            execution_time_seconds=execution_time_seconds,
            volume=volume_metrics,
            rhythm=rhythm_metrics,
            style=style_metrics,
            vocabulary=vocabulary_metrics,
            readability=readability_metrics,
            repetition=repetition_metrics,
            dialogue=dialogue_metrics,
        )


def _resolve_metrics(metrics: Sequence[MetricName] | Literal["all"]) -> frozenset[MetricName]:
    """Resolve the metrics argument to a frozenset of metric names.

    Args:
        metrics (Sequence[MetricName] | Literal["all"]): Metrics to calculate. Either 'all' or a sequence of metric
            names ("dialogue", "readability", "repetition", "rhythm", "style", "vocabulary", "volume").
            If an empty list is provided, all metrics will be computed.

    Returns:
        frozenset[MetricName]: A frozenset of metric names to compute.

    Raises:
        ValueError: If the metrics argument is invalid, i.e., not a valid metric name or sequence of metric names.
    """
    if isinstance(metrics, str):
        if metrics != "all":
            msg = f"Invalid metrics argument: {metrics!r}. Use 'all' or a sequence of metric names."
            raise ValueError(msg)
        selected_metrics = AVAILABLE_METRICS
    elif not metrics:
        selected_metrics = AVAILABLE_METRICS
    else:
        invalid_metrics = set(metrics) - AVAILABLE_METRICS
        if invalid_metrics:
            msg = f"Invalid metric(s): {sorted(invalid_metrics)}. Available metrics: {sorted(AVAILABLE_METRICS)}"
            raise ValueError(msg)
        selected_metrics = frozenset(metrics)

    return selected_metrics


def analyze(
    text: str,
    language: str = "en",
    model_name: str | None = None,
    doc: Doc | None = None,
    metrics: Sequence[MetricName] | Literal["all"] = "all",
    mattr_window_size: int = 100,
    msttr_segment_size: int = 100,
    words_per_minute: int = 200,
    short_threshold: int = 10,
    long_threshold: int = 30,
    use_lemmas: bool = True,
    repetition_window_size: int = 50,
) -> TextReport:
    """Convenience function to analyze text without explicitly creating a TextAnalyzer instance.

    Args:
        text (str): The raw text string to analyze.
        language (str): Target language code (ISO 639-1, e.g., "en" for English). Default is "en".
        model_name (str | None): Optional explicit spaCy model name.
        doc (Doc | None): Optional pre-parsed spaCy Doc object to bypass re-tokenization. If None, the text will be
            parsed using the spaCy pipeline.
        metrics (Sequence[MetricName] | Literal["all"]): Metrics to calculate. Either 'all' or a sequence of metric
            names ("dialogue", "readability", "repetition", "rhythm", "style", "vocabulary", "volume").
        mattr_window_size (int): Window size for MATTR calculation. Defaults to 100.
        msttr_segment_size (int): Segment size for MSTTR calculation. Defaults to 100.
        words_per_minute (int): Reading speed for reading time estimation. Defaults to 200.
        short_threshold (int): Upper word count bound for short sentences (< threshold). Defaults to 10 words.
        long_threshold (int): Lower word count bound for long sentences (> threshold). Defaults to 30 words.
        use_lemmas (bool): If True, uses normalized lemmas. If False, uses raw lower tokens. Defaults to True.
        repetition_window_size (int): Maximum distance, in content words, for two occurrences of the same word to be
            considered a close repetition. Defaults to 50.

    Returns:
        TextReport: A dataclass containing all computed metrics and analysis metadata.

    Raises:
        ValueError: If the metrics argument is invalid, i.e., not a valid metric name or sequence of metric names.
            Or if mattr_window_size, msttr_segment_size, repetition_window_size, or words_per_minute is less than 1.

    Examples:
        >>> report = analyze("The cat sat on the mat.")
        >>> report.volume.word_count
        6
        >>> report.readability.estimated_reading_time_minutes
        0.03
    """
    analyzer = TextAnalyzer(language=language, model_name=model_name)
    return analyzer.analyze(
        text=text,
        doc=doc,
        metrics=metrics,
        mattr_window_size=mattr_window_size,
        msttr_segment_size=msttr_segment_size,
        words_per_minute=words_per_minute,
        short_threshold=short_threshold,
        long_threshold=long_threshold,
        use_lemmas=use_lemmas,
        repetition_window_size=repetition_window_size,
    )


def pipe(
    texts: Iterable[str],
    language: str = "en",
    model_name: str | None = None,
    metrics: Sequence[MetricName] | Literal["all"] = "all",
    batch_size: int = 64,
    n_process: int = 1,
    mattr_window_size: int = 100,
    msttr_segment_size: int = 100,
    words_per_minute: int = 200,
    short_threshold: int = 10,
    long_threshold: int = 30,
    use_lemmas: bool = True,
    repetition_window_size: int = 50,
) -> Iterator[TextReport]:
    """Convenience function to analyze multiple texts without explicitly creating a TextAnalyzer instance.

    Texts are parsed through spaCy's batched pipe processing and reports are yielded lazily, in input order.
    See TextAnalyzer.pipe() for details on the amortized execution time stamping.

    Args:
        texts (Iterable[str]): The raw text strings to analyze. May be a generator; it is consumed lazily.
        language (str): Target language code (ISO 639-1, e.g., "en" for English). Default is "en".
        model_name (str | None): Optional explicit spaCy model name.
        metrics (Sequence[MetricName] | Literal["all"]): Metrics to calculate. Either 'all' or a sequence of metric
            names ("dialogue", "readability", "repetition", "rhythm", "style", "vocabulary", "volume").
        batch_size (int): Number of texts buffered per spaCy pipe batch. Defaults to 64.
        n_process (int): Number of processes for spaCy to use during parsing. Defaults to 1.
        mattr_window_size (int): Window size for MATTR calculation. Defaults to 100.
        msttr_segment_size (int): Segment size for MSTTR calculation. Defaults to 100.
        words_per_minute (int): Reading speed for reading time estimation. Defaults to 200.
        short_threshold (int): Upper word count bound for short sentences (< threshold). Defaults to 10 words.
        long_threshold (int): Lower word count bound for long sentences (> threshold). Defaults to 30 words.
        use_lemmas (bool): If True, uses normalized lemmas. If False, uses raw lower tokens. Defaults to True.
        repetition_window_size (int): Maximum distance, in content words, for two occurrences of the same word to be
            considered a close repetition. Defaults to 50.

    Returns:
        Iterator[TextReport]: A lazy iterator of reports, in the same order as the input texts.

    Raises:
        ValueError: if the metrics argument is invalid, or if batch_size or n_process is less than 1.

    Examples:
        >>> reports = list(pipe(["The cat sat.", "A dog runs fast."], language="en"))
        >>> [r.volume.word_count for r in reports]
        [4, 4]
    """
    analyzer = TextAnalyzer(language=language, model_name=model_name)
    return analyzer.pipe(
        texts=texts,
        metrics=metrics,
        batch_size=batch_size,
        n_process=n_process,
        mattr_window_size=mattr_window_size,
        msttr_segment_size=msttr_segment_size,
        words_per_minute=words_per_minute,
        short_threshold=short_threshold,
        long_threshold=long_threshold,
        use_lemmas=use_lemmas,
        repetition_window_size=repetition_window_size,
    )
