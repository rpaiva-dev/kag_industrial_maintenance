"""Centralized client for the LLM (OpenAI).

Why centralize? Every module (extraction, entity identification, answer
generation) needs to call the LLM. Concentrating client creation, key
loading and the default model in one place avoids duplication and makes it
easy to swap model or provider later — every other module only knows about
the call_llm() function.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

# Project default model. A single place to change it when needed.
DEFAULT_MODEL = "gpt-4o"

_client: OpenAI | None = None


def _get_api_key() -> str | None:
    """Resolve the API key: local .env first, then st.secrets (production).

    The streamlit import is done lazily (inside the function) because the
    offline scripts (extraction, graph_builder) run outside Streamlit and
    shouldn't depend on it.
    """
    load_dotenv()  # loads .env if it exists; doesn't override already-set vars
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st

        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


def get_client() -> OpenAI:
    """Return a singleton client. Fails early, with a clear message, if there's no key."""
    global _client
    if _client is None:
        key = _get_api_key()
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. Create a .env file "
                "(see .env.example) or configure st.secrets."
            )
        _client = OpenAI(api_key=key)
    return _client


def call_llm(
    system: str,
    user: str,
    max_tokens: int = 4096,
    json_schema: dict | None = None,
) -> str:
    """Make a simple call to the LLM and return the response text.

    json_schema: when provided, uses structured outputs (response_format
    json_schema with strict=True) to GUARANTEE the response is valid JSON
    matching the schema — more robust than asking for JSON in the prompt
    and hoping for the best.
    """
    client = get_client()

    kwargs: dict = dict(
        model=DEFAULT_MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    if json_schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "output", "strict": True, "schema": json_schema},
        }

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""
