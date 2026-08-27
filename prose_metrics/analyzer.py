"""Central orchestrator for prose and text analysis."""

import time
from collections.abc import Sequence
from typing import Final, Literal

from spacy.tokens import Doc

from prose_metrics.metrics import (
    compute_readability_metrics,
    compute_rhythm_metrics,
    compute_style_metrics,
    compute_vocabulary_metrics,
    compute_volume_metrics,
)
from prose_metrics.models.report import (
    ReadabilityMetrics,
    RhythmMetrics,
    StyleMetrics,
    TextReport,
    VocabularyMetrics,
    VolumeMetrics,
)
from prose_metrics.nlp.pipeline import SpacyPipelineManager

MetricName = Literal["readability", "rhythm", "style", "vocabulary", "volume"]

AVAILABLE_METRICS: Final[frozenset[MetricName]] = frozenset({"readability", "rhythm", "style", "vocabulary", "volume"})


class TextAnalyzer:
    """Main analyzer coordinating NLP parsing and metric computations."""

    def __init__(
        self,
        language: str = "en",
        model_name: str | None = None,
    ) -> None:
        """Initialize the analyzer with a specific language of model.

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
        words_per_minute: int = 200,
    ) -> TextReport:
        """Analyze a given text and generate a structure TextReport.

        Args:
            text (str): The raw text string to analyze.
            doc (Doc | None): Optional pre-parsed spaCy Doc object to bypass re-tokenization. If None, the text will be
                parsed using the spaCy pipeline.
            metrics (Sequence[MetricName] | Literal["all"]): Metrics to calculate. Either 'all' or a sequence of metric
                names ("readability", "rhythm", "style", "vocabulary", "volume"). If an empty list is provided,
                all metrics will be computed.
            mattr_window_size (int): Window size for MATTR calculation. Defaults to 100.
            words_per_minute (int): Reading speed for reading time estimation. Defaults to 200.

        Returns:
            TextReport: A dataclass containing all computed metrics and analysis metadata.

        Raises:
            ValueError: If an unknown metric name is provided.
        """
        start_time = time.perf_counter()

        # Resolve metrics to compute
        if metrics == "all" or not metrics:
            selected_metrics = AVAILABLE_METRICS
        else:
            invalid_metrics = set(metrics) - AVAILABLE_METRICS
            if invalid_metrics:
                msg = f"Invalid metric(s): {sorted(invalid_metrics)}. Available metrics: {sorted(AVAILABLE_METRICS)}"
                raise ValueError(msg)
            selected_metrics = frozenset(metrics)

        # Parse text with spaCy (if not already provided)
        if doc is None:
            nlp = self._pipeline_manager.get_pipeline(language=self.language, model_name=self.model_name)
            parsed_doc = nlp(text=text)
        else:
            parsed_doc = doc

        # Compute selected metrics
        volume_metrics: VolumeMetrics | None = None
        rhythm_metrics: RhythmMetrics | None = None
        style_metrics: StyleMetrics | None = None
        vocabulary_metrics: VocabularyMetrics | None = None
        readability_metrics: ReadabilityMetrics | None = None

        if "volume" in selected_metrics:
            volume_metrics = compute_volume_metrics(text=text, doc=parsed_doc)

        if "rhythm" in selected_metrics:
            rhythm_metrics = compute_rhythm_metrics(doc=parsed_doc)

        if "style" in selected_metrics:
            style_metrics = compute_style_metrics(doc=parsed_doc)

        if "vocabulary" in selected_metrics:
            vocabulary_metrics = compute_vocabulary_metrics(doc=parsed_doc, mattr_window_size=mattr_window_size)

        if "readability" in selected_metrics:
            readability_metrics = compute_readability_metrics(
                text=text,
                doc=parsed_doc,
                language=self.language,
                words_per_minute=words_per_minute,
            )

        execution_time = round(time.perf_counter() - start_time, 4)

        return TextReport(
            language=self.language,
            execution_time_seconds=execution_time,
            volume=volume_metrics,
            rhythm=rhythm_metrics,
            style=style_metrics,
            vocabulary=vocabulary_metrics,
            readability=readability_metrics,
        )


def analyze(
    text: str,
    language: str = "en",
    model_name: str | None = None,
    doc: Doc | None = None,
    metrics: Sequence[MetricName] | Literal["all"] = "all",
    mattr_window_size: int = 100,
    words_per_minute: int = 200,
) -> TextReport:
    """Convenience function to analyze text without explicitly creating a TextAnalyzer instance.

    Args:
        text (str): The raw text string to analyze.
        language (str): Target language code (ISO 639-1, e.g., "en" for English). Default is "en".
        model_name (str | None): Optional explicit spaCy model name.
        doc (Doc | None): Optional pre-parsed spaCy Doc object to bypass re-tokenization. If None, the text will be
            parsed using the spaCy pipeline.
        metrics (Sequence[MetricName] | Literal["all"]): Metrics to calculate. Either 'all' or a sequence of metric
            names ("readability", "rhythm", "style", "vocabulary", "volume").
        mattr_window_size (int): Window size for MATTR calculation. Defaults to 100.
        words_per_minute (int): Reading speed for reading time estimation. Defaults to 200.

    Returns:
        TextReport: A dataclass containing all computed metrics and analysis metadata.

    Raises:
        ValueError: If an unknown metric name is provided.
    """
    analyzer = TextAnalyzer(language=language, model_name=model_name)
    return analyzer.analyze(
        text=text,
        doc=doc,
        metrics=metrics,
        mattr_window_size=mattr_window_size,
        words_per_minute=words_per_minute,
    )
