"""
HuggingFace model management and discovery.

This module provides utilities to search and list available TTS models
from the HuggingFace Hub using the huggingface-hub library.
"""

from typing import List, Dict, Optional
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError


class HuggingFaceModelManager:
    """Manager for discovering and managing HuggingFace TTS models."""

    # Default TTS model
    DEFAULT_MODEL = "espnet/eng_male_fgl"

    def __init__(self, token: Optional[str] = None):
        self._api = HfApi(token=token)

    def list_tts_models(
        self,
        task: str = "text-to-speech",
        limit: int = 20,
        sort: str = "likes",
    ) -> List[Dict]:
        """
        Fetch available TTS models from HuggingFace Hub.

        Args:
            task: The task type to filter by (default: "text-to-speech")
            limit: Maximum number of models to return (default: 20)
            sort: Sort order - "likes", "downloads", "trending" (default: "likes")

        Returns:
            List of dictionaries with model information:
            - id: Model ID (e.g., "espnet/eng_male_fgl")
            - name: Model name
            - downloads: Number of downloads
            - likes: Number of likes
            - pipeline_tag: Task type
            - description: Model description (if available)

        Raises:
            HfHubHTTPError: If API call fails
        """
        sort_map = {
            "likes": "likes",
            "downloads": "downloads",
            "trending": "trending",
        }

        models = self._api.list_models(
            task=task,
            sort=sort_map.get(sort, "likes"),
            direction=-1,  # descending
            limit=limit,
        )

        result = []
        for model in models:
            model_info = {
                "id": model.modelId,
                "name": (
                    model.modelId.split("/")[-1]
                    if "/" in model.modelId
                    else model.modelId
                ),
                "downloads": model.downloads or 0,
                "likes": model.likes or 0,
                "pipeline_tag": model.pipeline_tag or "",
            }
            if model.description:
                model_info["description"] = model.description
            result.append(model_info)

        return result

    def search_models(
        self,
        query: str,
        task: str = "text-to-speech",
        limit: int = 10,
    ) -> List[Dict]:
        """
        Search for TTS models on HuggingFace Hub by keyword.

        Args:
            query: Search query (e.g., "hindi", "female", "english")
            task: The task type to filter by (default: "text-to-speech")
            limit: Maximum number of results to return (default: 10)

        Returns:
            List of matching models with same structure as list_tts_models()

        Raises:
            HfHubHTTPError: If API call fails
        """
        models = self._api.list_models(
            task=task,
            search=query,
            limit=limit,
        )

        result = []
        for model in models:
            if model.pipeline_tag == task:
                model_info = {
                    "id": model.modelId,
                    "name": (
                        model.modelId.split("/")[-1]
                        if "/" in model.modelId
                        else model.modelId
                    ),
                    "downloads": model.downloads or 0,
                    "likes": model.likes or 0,
                    "pipeline_tag": model.pipeline_tag or "",
                }
                if model.description:
                    model_info["description"] = model.description
                result.append(model_info)

        return result

    @staticmethod
    def format_models_table(models: List[Dict]) -> str:
        """
        Format models list as a readable table.

        Args:
            models: List of model dictionaries

        Returns:
            Formatted table string for display
        """
        if not models:
            return "No models found."

        # Calculate column widths
        id_width = max(
            len("Model ID"), max(len(m["id"]) for m in models)
        )
        downloads_width = max(
            len("Downloads"),
            max(len(str(m.get("downloads", 0))) for m in models),
        )
        likes_width = max(
            len("Likes"),
            max(len(str(m.get("likes", 0))) for m in models),
        )

        # Build table
        lines = []
        header = (
            f"{'Model ID':<{id_width}}"
            f" | {'Downloads':>{downloads_width}}"
            f" | {'Likes':>{likes_width}}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for model in models:
            line = (
                f"{model['id']:<{id_width}}"
                f" | {model.get('downloads', 0):>{downloads_width}}"
                f" | {model.get('likes', 0):>{likes_width}}"
            )
            lines.append(line)

        return "\n".join(lines)
