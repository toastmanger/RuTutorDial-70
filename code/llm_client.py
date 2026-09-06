"""Generic OpenAI-compatible async client for the PUBLISHED artifact.

Drop-in replacement for the internal gateway client used during the study.
Configuration via environment only — no organization-specific endpoints:

  LLM_BASE_URL   e.g. https://api.openai.com/v1 | https://openrouter.ai/api/v1 | http://localhost:8000/v1
  LLM_API_KEY    provider key

Usage in scripts (artifact build replaces `from llm_client import make_client`):
  from generic_client import make_client
"""
from __future__ import annotations

import os

from openai import AsyncOpenAI


def make_client(model: str) -> AsyncOpenAI:  # model kept for interface parity
    base_url = os.environ.get("LLM_BASE_URL")
    api_key = os.environ.get("LLM_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError("Set LLM_BASE_URL and LLM_API_KEY environment variables")
    return AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)


def normalize_model_id(model: str) -> str:
    return model
