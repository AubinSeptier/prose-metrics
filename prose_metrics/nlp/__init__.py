"""NLP module providing optimized linguistic pipelines for text analysis tasks."""

from prose_metrics.nlp.exceptions import ModelNotFoundError, NLPError
from prose_metrics.nlp.pipeline import SpacyPipelineManager

__all__ = ["NLPError", "ModelNotFoundError", "SpacyPipelineManager"]
