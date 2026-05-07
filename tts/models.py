"""
HuggingFace model management and discovery.

This module provides utilities to search and list available TTS models
from the HuggingFace Hub using the Hugging Face Hub API.
"""

import requests
from typing import List, Dict, Optional


class HuggingFaceModelManager:
    """Manager for discovering and managing HuggingFace TTS models."""
    
    # Default TTS model
    DEFAULT_MODEL = "espnet/eng_male_fgl"
    
    # HuggingFace API endpoint
    HF_API_BASE = "https://huggingface.co/api"
    
    @staticmethod
    def list_tts_models(
        task: str = "text-to-speech",
        limit: int = 20,
        sort: str = "likes"
    ) -> List[Dict[str, str]]:
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
            requests.RequestException: If API call fails
        """
        try:
            # Use HuggingFace Hub API to search for models
            url = f"{HuggingFaceModelManager.HF_API_BASE}/models"
            params = {
                "task": task,
                "sort": sort,
                "limit": limit,
                "library": "transformers"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            models_data = response.json()
            
            # Transform API response into readable format
            models = []
            for model in models_data:
                if model.get("pipeline_tag") == "text-to-speech":
                    model_info = {
                        "id": model.get("id", ""),
                        "name": model.get("id", "").split("/")[-1] if "/" in model.get("id", "") else model.get("id", ""),
                        "downloads": model.get("downloads", 0),
                        "likes": model.get("likes", 0),
                        "pipeline_tag": model.get("pipeline_tag", ""),
                    }
                    if model.get("description"):
                        model_info["description"] = model.get("description")
                    models.append(model_info)
            
            return models
        
        except requests.exceptions.Timeout:
            raise requests.RequestException("Request to HuggingFace API timed out")
        except requests.exceptions.ConnectionError:
            raise requests.RequestException("Failed to connect to HuggingFace API")
        except ValueError as e:
            raise requests.RequestException(f"Invalid JSON response from HuggingFace API: {str(e)}")
    
    @staticmethod
    def search_models(
        query: str,
        task: str = "text-to-speech",
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """
        Search for TTS models on HuggingFace Hub by keyword.
        
        Args:
            query: Search query (e.g., "hindi", "female", "english")
            task: The task type to filter by (default: "text-to-speech")
            limit: Maximum number of results to return (default: 10)
        
        Returns:
            List of matching models with same structure as list_tts_models()
        
        Raises:
            requests.RequestException: If API call fails
        """
        try:
            url = f"{HuggingFaceModelManager.HF_API_BASE}/models"
            params = {
                "search": query,
                "task": task,
                "limit": limit,
                "library": "transformers"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            models_data = response.json()
            
            # Transform API response
            models = []
            for model in models_data:
                if model.get("pipeline_tag") == "text-to-speech":
                    model_info = {
                        "id": model.get("id", ""),
                        "name": model.get("id", "").split("/")[-1] if "/" in model.get("id", "") else model.get("id", ""),
                        "downloads": model.get("downloads", 0),
                        "likes": model.get("likes", 0),
                        "pipeline_tag": model.get("pipeline_tag", ""),
                    }
                    if model.get("description"):
                        model_info["description"] = model.get("description")
                    models.append(model_info)
            
            return models
        
        except requests.exceptions.Timeout:
            raise requests.RequestException("Request to HuggingFace API timed out")
        except requests.exceptions.ConnectionError:
            raise requests.RequestException("Failed to connect to HuggingFace API")
        except ValueError as e:
            raise requests.RequestException(f"Invalid JSON response from HuggingFace API: {str(e)}")
    
    @staticmethod
    def format_models_table(models: List[Dict[str, str]]) -> str:
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
        id_width = max(len("Model ID"), max(len(m["id"]) for m in models))
        downloads_width = max(len("Downloads"), max(len(str(m.get("downloads", 0))) for m in models))
        likes_width = max(len("Likes"), max(len(str(m.get("likes", 0))) for m in models))
        
        # Build table
        lines = []
        header = f"{'Model ID':<{id_width}} | {'Downloads':>{downloads_width}} | {'Likes':>{likes_width}}"
        lines.append(header)
        lines.append("-" * len(header))
        
        for model in models:
            line = f"{model['id']:<{id_width}} | {model.get('downloads', 0):>{downloads_width}} | {model.get('likes', 0):>{likes_width}}"
            lines.append(line)
        
        return "\n".join(lines)
