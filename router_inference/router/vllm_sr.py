# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
vLLM Semantic Router implementation for RouterArena.

This router calls the vllm-project/semantic-router classification API
to determine the category of a query, then maps it to one of the
configured models.
"""

import json
import urllib.request
import urllib.error
from typing import Dict, Optional

from router_inference.router.base_router import BaseRouter


class VLLMSR(BaseRouter):
    """
    vLLM Semantic Router implementation.

    Uses the vllm-sr classification API to categorize queries and route
    them to the most appropriate model based on the detected category.
    """

    # Category to model mapping - consolidated for stability
    # Analysis showed gemini-2.0-flash-001 had only 16.9% stability
    CATEGORY_MODEL_MAPPING: Dict[str, str] = {
        "biology": "claude-3-haiku-20240307",
        "business": "gemini-2.0-flash-001",
        "chemistry": "deepseek-chat",
        "computer science": "gemini-2.0-flash-001",
        "economics": "gemini-2.0-flash-001",
        "engineering": "deepseek-chat",
        "health": "gpt-4o-mini",
        "history": "gpt-4o-mini",
        "law": "gemini-2.0-flash-001",
        "math": "gemini-2.0-flash-001",
        "other": "gemini-2.0-flash-001",
        "philosophy": "gemini-2.0-flash-001",
        "physics": "gemini-2.0-flash-001",
        "psychology": "gemini-2.0-flash-001",
    }

    DEFAULT_MODEL = "gpt-4o-mini"
    API_URL = "http://localhost:8080/api/v1/classify/intent"

    def __init__(self, router_name: str):
        super().__init__(router_name)

    def _call_classify_api(self, query: str) -> Optional[Dict]:
        """Call the vllm-sr classification API."""
        try:
            data = json.dumps({"text": query}).encode("utf-8")
            req = urllib.request.Request(
                self.API_URL,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"  Warning: Classification API failed: {e}")
            return None

    def _get_prediction(self, query: str) -> str:
        """Get the model prediction using vllm-sr classification."""
        result = self._call_classify_api(query)

        if result and "classification" in result:
            category = result["classification"].get("category", "other")
            model = self.CATEGORY_MODEL_MAPPING.get(category, self.DEFAULT_MODEL)
            if model in self.models:
                return model

        return self.DEFAULT_MODEL if self.DEFAULT_MODEL in self.models else self.models[0]
