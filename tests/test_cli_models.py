"""Tests for CLI model listing commands."""

import sys
import pytest
from unittest.mock import patch, MagicMock
from tts.tts import list_hf_models
from huggingface_hub.utils import HfHubHTTPError


class TestListHFModelsCLI:
    """Tests for the list_hf_models CLI function."""

    @patch("tts.tts.HuggingFaceModelManager")
    def test_list_hf_models_success(self, MockManager):
        """Test that list_hf_models prints a table of models."""
        mock_instance = MockManager.return_value
        mock_instance.list_tts_models.return_value = [
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

        with patch("builtins.print") as mock_print:
            list_hf_models(limit=20)

            # Verify the table was printed
            mock_print.assert_any_call(
                "\nFetching top 20 HuggingFace TTS models...\n"
            )

            # Verify the manager was called correctly
            mock_instance.list_tts_models.assert_called_once_with(limit=20)

    @patch("tts.tts.HuggingFaceModelManager")
    def test_search_hf_models_success(self, MockManager):
        """Test that search_hf_models prints a table of matching models."""
        mock_instance = MockManager.return_value
        mock_instance.search_models.return_value = [
            {
                "id": "espnet/hindi_male_fgl",
                "name": "hindi_male_fgl",
                "downloads": 54321,
                "likes": 432,
                "pipeline_tag": "text-to-speech",
                "description": "Hindi male voice",
            },
        ]

        with patch("builtins.print") as mock_print:
            list_hf_models(query="hindi", limit=10)

            # Verify search message
            mock_print.assert_any_call(
                "\nSearching for HuggingFace TTS models matching 'hindi'...\n"
            )

            # Verify the search was called correctly
            mock_instance.search_models.assert_called_once_with(
                query="hindi", limit=10
            )

    @patch("tts.tts.HuggingFaceModelManager")
    def test_list_hf_models_no_results(self, MockManager):
        """Test list_hf_models with no results."""
        mock_instance = MockManager.return_value
        mock_instance.list_tts_models.return_value = []

        with patch("builtins.print") as mock_print:
            list_hf_models(limit=20)
            mock_print.assert_any_call("No TTS models found.")

    @patch("tts.tts.HuggingFaceModelManager")
    def test_search_hf_models_no_results(self, MockManager):
        """Test search_hf_models with no results."""
        mock_instance = MockManager.return_value
        mock_instance.search_models.return_value = []

        with patch("builtins.print") as mock_print:
            list_hf_models(query="nonexistent", limit=10)
            mock_print.assert_any_call("No TTS models found.")
            mock_print.assert_any_call("Try a different search query.")

    @patch("tts.tts.HuggingFaceModelManager")
    def test_list_hf_models_api_error(self, MockManager):
        """Test that API errors are handled gracefully."""
        mock_instance = MockManager.return_value
        mock_instance.list_tts_models.side_effect = HfHubHTTPError(
            "API Error", response=MagicMock(status_code=500)
        )

        with pytest.raises(SystemExit) as exc_info:
            list_hf_models(limit=20)

        assert exc_info.value.code == 1
