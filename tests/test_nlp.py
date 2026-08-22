"""Unit tests for the NLP pipeline module."""

from unittest.mock import MagicMock, patch

import pytest
from spacy.language import Language

from prose_metrics.nlp.exceptions import ModelNotFoundError
from prose_metrics.nlp.pipeline import SpacyPipelineManager


@pytest.fixture(autouse=True)
def reset_manager() -> None:
    """Reset manager singleton state before each test."""
    manager = SpacyPipelineManager()
    manager.clear_cache()


def test_singleton_identity() -> None:
    """Check that SpacyPipelineManager is a singleton."""
    manager_a = SpacyPipelineManager()
    manager_b = SpacyPipelineManager()
    assert manager_a is manager_b


def test_unsupported_language_raises_value_error() -> None:
    """Check ValueError on unknown language without custom model."""
    manager = SpacyPipelineManager()
    with pytest.raises(ValueError, match=r"Unsupported language code: 'an'."):
        manager.get_pipeline(language="an")


@patch("spacy.load")
def test_missing_model_raises_model_not_found_error(mock_spacy_load: MagicMock) -> None:
    """Check ModelNotFoundError is raised when spacy.load fails with OSError."""
    mock_spacy_load.side_effect = OSError("Model not found")
    manager = SpacyPipelineManager()

    with pytest.raises(ModelNotFoundError) as exc_info:
        manager.get_pipeline(language="en")

    assert "en_core_web_sm" in str(exc_info.value)
    assert exc_info.value.language == "en"
    assert exc_info.value.model_name == "en_core_web_sm"


def test_pipeline_caching_and_disable() -> None:
    """Check pipeline is cached and loaded with disabled components."""
    manager = SpacyPipelineManager()
    mock_nlp = MagicMock(spec=Language)

    with patch("spacy.load", return_value=mock_nlp) as mock_load:
        # First call loads model
        nlp1 = manager.get_pipeline(language="en")
        assert nlp1 is mock_nlp
        mock_load.assert_called_once_with("en_core_web_sm", disable=["ner"])

        # Second call returns cached instance without reloading
        nlp2 = manager.get_pipeline(language="en")
        assert nlp2 is mock_nlp
        assert mock_load.call_count == 1


@patch("spacy.load")
def test_explicit_model_name_overrides_default(mock_spacy_load: MagicMock) -> None:
    """Check that an explicit model_name bypasses the language default."""
    mock_spacy_load.return_value = MagicMock(spec=Language)
    manager = SpacyPipelineManager()

    nlp = manager.get_pipeline(language="en", model_name="en_core_web_md")

    assert nlp is mock_spacy_load.return_value
    mock_spacy_load.assert_called_once_with("en_core_web_md", disable=["ner"])


@patch("spacy.load")
def test_custom_disable_creates_distinct_cache_entry(mock_spacy_load: MagicMock) -> None:
    """Check that different disable tuples produce separate cached pipelines."""
    mock_spacy_load.side_effect = [MagicMock(spec=Language), MagicMock(spec=Language)]
    manager = SpacyPipelineManager()

    nlp_default = manager.get_pipeline(language="en")
    nlp_custom = manager.get_pipeline(language="en", disable=("ner", "lemmatizer"))

    assert nlp_default is not nlp_custom
    assert mock_spacy_load.call_count == 2
    mock_spacy_load.assert_any_call("en_core_web_sm", disable=["ner"])
    mock_spacy_load.assert_any_call("en_core_web_sm", disable=["ner", "lemmatizer"])


@patch("spacy.load")
def test_clear_cache_forces_reload(mock_spacy_load: MagicMock) -> None:
    """Check that clear_cache empties the cache and forces a reload."""
    mock_spacy_load.return_value = MagicMock(spec=Language)
    manager = SpacyPipelineManager()

    manager.get_pipeline(language="en")
    manager.clear_cache()
    manager.get_pipeline(language="en")

    assert mock_spacy_load.call_count == 2
