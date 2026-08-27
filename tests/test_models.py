"""Tests for HuggingFace model listing functionality."""

import pytest
from unittest.mock import patch, MagicMock
from tts.models import HuggingFaceModelManager


class TestHuggingFaceModelManager:
    """Tests for HuggingFaceModelManager."""

    @pytest.fixture
    def mock_hf_model(self):
        """Create a mock HuggingFace model object."""
        model = MagicMock()
        model.modelId = "espnet/eng_male_fgl"
        model.downloads = 123456
        model.likes = 987
        model.pipeline_tag = "text-to-speech"
        model.description = "English male voice"
        return model

    @pytest.fixture
    def mock_hf_model_no_desc(self):
        """Create a mock HuggingFace model without description."""
        model = MagicMock()
        model.modelId = "facebook/mms-tts-eng"
        model.downloads = 234567
        model.likes = 876
        model.pipeline_tag = "text-to-speech"
        model.description = None
        return model

    def test_list_tts_models(self, mock_hf_model, mock_hf_model_no_desc):
        """Test that list_tts_models returns correctly formatted model data."""
        with patch("tts.models.HfApi") as MockHfApi:
            mock_api_instance = MockHfApi.return_value
            mock_api_instance.list_models.return_value = [
                mock_hf_model,
                mock_hf_model_no_desc,
            ]

            manager = HuggingFaceModelManager()
            models = manager.list_tts_models(limit=5)

            assert len(models) == 2

            # Check first model
            assert models[0]["id"] == "espnet/eng_male_fgl"
            assert models[0]["name"] == "eng_male_fgl"
            assert models[0]["downloads"] == 123456
            assert models[0]["likes"] == 987
            assert models[0]["pipeline_tag"] == "text-to-speech"
            assert models[0]["description"] == "English male voice"

            # Check second model (no description)
            assert models[1]["id"] == "facebook/mms-tts-eng"
            assert "description" not in models[1]

            # Verify API was called with correct parameters
            mock_api_instance.list_models.assert_called_once_with(
                task="text-to-speech",
                sort="likes",
                direction=-1,
                limit=5,
            )

    def test_search_models(self, mock_hf_model):
        """Test that search_models returns correctly filtered model data."""
        with patch("tts.models.HfApi") as MockHfApi:
            mock_api_instance = MockHfApi.return_value
            mock_api_instance.list_models.return_value = [mock_hf_model]

            manager = HuggingFaceModelManager()
            models = manager.search_models(query="english", limit=10)

            assert len(models) == 1
            assert models[0]["id"] == "espnet/eng_male_fgl"

            # Verify API was called with search parameter
            mock_api_instance.list_models.assert_called_once_with(
                task="text-to-speech",
                search="english",
                limit=10,
            )

    def test_search_models_filters_wrong_task(self, mock_hf_model):
        """Test that search_models filters out models that don't match the task."""
        with patch("tts.models.HfApi") as MockHfApi:
            # Create a model that has a different pipeline_tag
            wrong_model = MagicMock()
            wrong_model.modelId = "some/other-model"
            wrong_model.pipeline_tag = "text-generation"  # wrong task
            wrong_model.downloads = 0
            wrong_model.likes = 0
            wrong_model.description = None

            mock_api_instance = MockHfApi.return_value
            mock_api_instance.list_models.return_value = [mock_hf_model, wrong_model]

            manager = HuggingFaceModelManager()
            models = manager.search_models(query="test")

            # Only the TTS model should be included
            assert len(models) == 1
            assert models[0]["id"] == "espnet/eng_male_fgl"

    def test_list_tts_models_empty(self):
        """Test that list_tts_models handles empty results."""
        with patch("tts.models.HfApi") as MockHfApi:
            mock_api_instance = MockHfApi.return_value
            mock_api_instance.list_models.return_value = []

            manager = HuggingFaceModelManager()
            models = manager.list_tts_models(limit=20)

            assert models == []

    def test_format_models_table(self, mock_hf_model, mock_hf_model_no_desc):
        """Test that format_models_table produces correctly formatted output."""
        models_data = [
            {
                "id": "espnet/eng_male_fgl",
                "name": "eng_male_fgl",
                "downloads": 123456,
                "likes": 987,
                "pipeline_tag": "text-to-speech",
                "description": "English male voice",
            },
            {
                "id": "facebook/mms-tts-eng",
                "name": "mms-tts-eng",
                "downloads": 234567,
                "likes": 876,
                "pipeline_tag": "text-to-speech",
            },
        ]

        table = HuggingFaceModelManager.format_models_table(models_data)
        lines = table.split("\n")

        # Check header
        assert "Model ID" in lines[0]
        assert "Downloads" in lines[0]
        assert "Likes" in lines[0]

        # Check separator
        assert lines[1].startswith("---")

        # Check model rows
        assert "espnet/eng_male_fgl" in lines[2]
        assert "123456" in lines[2]
        assert "987" in lines[2]

        assert "facebook/mms-tts-eng" in lines[3]
        assert "234567" in lines[3]
        assert "876" in lines[3]

    def test_format_models_table_empty(self):
        """Test that format_models_table handles empty input."""
        table = HuggingFaceModelManager.format_models_table([])
        assert table == "No models found."

    def test_default_model_constant(self):
        """Test that the default model constant is set correctly."""
        assert HuggingFaceModelManager.DEFAULT_MODEL == "espnet/eng_male_fgl"
