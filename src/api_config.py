"""Shared configuration for the OpenAI-compatible Qwen endpoint."""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_DIR / ".env")

QWEN_BASE_URL = os.environ.get(
    "QWEN_BASE_URL",
    "https://spark-da32.tail67be05.ts.net:8443/v1",
)
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3.6:35b")


def get_qwen_api_key():
    """Return the API key without keeping a secret in the source code."""

    api_key = os.environ.get("QWEN_API_KEY")
    if not api_key or api_key == "replace-with-your-api-key":
        raise RuntimeError(
            "QWEN_API_KEY is not configured. Copy .env.example to .env, "
            "then add your API key to .env."
        )

    return api_key


def create_qwen_client(**options):
    """Create a client using the project-wide endpoint configuration."""

    return OpenAI(
        base_url=QWEN_BASE_URL,
        api_key=get_qwen_api_key(),
        **options,
    )
