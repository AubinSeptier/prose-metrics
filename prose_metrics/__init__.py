"""This module serves as the entry point for the prose-metrics package."""

from importlib.metadata import PackageNotFoundError, version

from prose_metrics.analyzer import TextAnalyzer, analyze
from prose_metrics.models.report import (
    ReadabilityMetrics,
    RepetitionMetrics,
    RhythmMetrics,
    StyleMetrics,
    TextReport,
    VocabularyMetrics,
    VolumeMetrics,
)
from prose_metrics.nlp.exceptions import ModelNotFoundError, NLPError
from prose_metrics.nlp.pipeline import SpacyPipelineManager

try:
    __version__ = version("prose-metrics")
except PackageNotFoundError:
    __version__ = "0.1.0"  # Package is not installed

__all__ = [
    "TextAnalyzer",
    "analyze",
    "ReadabilityMetrics",
    "RepetitionMetrics",
    "RhythmMetrics",
    "StyleMetrics",
    "TextReport",
    "VocabularyMetrics",
    "VolumeMetrics",
    "ModelNotFoundError",
    "NLPError",
    "SpacyPipelineManager",
    "__version__",
]
