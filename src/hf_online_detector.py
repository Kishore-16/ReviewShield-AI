"""
Hugging Face Online Detector Module (Temporary / Pluggable Module)
File: src/hf_online_detector.py

This module queries Hugging Face's Serverless Inference API for AI-generated text detection.
Can be deleted at any time without breaking the core local ML pipeline.
"""

import os
import requests
from pathlib import Path
from typing import Dict, Any, Optional

DEFAULT_HF_MODEL = "Hello-SimpleAI/chatgpt-detector-roberta"
HF_ROUTER_URL = "https://router.huggingface.co/hf-inference/models"

def _load_env_file():
    """Helper to load environment variables from .env file if present."""
    root_dir = Path(__file__).resolve().parent.parent
    possible_envs = [
        root_dir / ".env",
        root_dir / "tests" / ".env",
        Path.cwd() / ".env"
    ]
    for env_path in possible_envs:
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, val = line.split("=", 1)
                            key_str = key.strip()
                            if key_str not in os.environ:
                                os.environ[key_str] = val.strip().strip("'\"")
            except Exception:
                pass

class HFOnlineDetector:
    """Hugging Face Inference API client for AI review detection."""

    def __init__(self, api_token: Optional[str] = None, model_name: str = DEFAULT_HF_MODEL):
        _load_env_file()
        self.model_name = model_name
        self.api_token = api_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
        self.endpoint_url = f"{HF_ROUTER_URL}/{self.model_name}"

    def is_available(self) -> bool:
        """Returns True if an API token is present and ready for online calls."""
        return bool(self.api_token)

    def predict_online(self, text: str, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Sends text to Hugging Face Inference API.
        Returns detailed probability dictionary or graceful fallback if unauthenticated / offline.
        """
        if not self.api_token:
            return {
                "online_enabled": False,
                "status": "skipped",
                "message": "HF_TOKEN / HUGGINGFACE_API_KEY environment variable not set. Using local ML model fallback.",
                "human_confidence": None,
                "ai_confidence": None
            }

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        payload = {"inputs": text}

        try:
            response = requests.post(self.endpoint_url, headers=headers, json=payload, timeout=timeout)
            
            if response.status_code == 200:
                data = response.json()
                # Expected format: [[{'label': 'Human', 'score': 0.98}, {'label': 'ChatGPT', 'score': 0.02}]]
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                    scores = {item['label'].lower(): item['score'] for item in data[0]}
                    human_score = scores.get('human', scores.get('label_0', 0.5)) * 100
                    ai_score = scores.get('chatgpt', scores.get('cg', scores.get('label_1', 0.5))) * 100
                    
                    return {
                        "online_enabled": True,
                        "status": "success",
                        "model_name": self.model_name,
                        "human_confidence": round(human_score, 2),
                        "ai_confidence": round(ai_score, 2),
                        "is_ai": ai_score > human_score,
                        "raw": data
                    }
                else:
                    return {
                        "online_enabled": True,
                        "status": "error",
                        "message": f"Unexpected response format: {data}",
                        "human_confidence": None,
                        "ai_confidence": None
                    }
            elif response.status_code == 503:
                return {
                    "online_enabled": True,
                    "status": "loading",
                    "message": "Hugging Face model is currently loading. Falling back to local model.",
                    "human_confidence": None,
                    "ai_confidence": None
                }
            else:
                return {
                    "online_enabled": True,
                    "status": "error",
                    "message": f"HF API returned status {response.status_code}: {response.text[:150]}",
                    "human_confidence": None,
                    "ai_confidence": None
                }

        except Exception as e:
            return {
                "online_enabled": True,
                "status": "exception",
                "message": f"Online inference request failed: {str(e)}",
                "human_confidence": None,
                "ai_confidence": None
            }
