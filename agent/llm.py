"""Calls Claude through the Anthropic Messages API."""
from __future__ import annotations

import json
import os
import re
import time

import requests

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-5-5"


class LLMError(RuntimeError):
    pass


def ask(system: str, user: str, *, max_tokens: int = 4000, api_key: str | None = None,
        model: str | None = None, session: requests.Session | None = None) -> str:
    # Add your code here to handle the new requirement
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise LLMError("ANTHROPIC_API_KEY is not set")
    s = session or requests
    body = {
        "model": model or os.environ.get("ADT_MODEL") or DEFAULT_MODEL,  # empty var -> default
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    for attempt in range(4):
        r = s.post(API_URL, headers=headers, json=body, timeout=180)
        if r.status_code in (429, 500, 502, 503, 529) and attempt < 3:
            time.sleep(5 * (attempt + 1))
            continue
        break
    if r.status_code != 200:
        raise LLMError(f"Claude API {r.status_code}: {r.text[:300]}")
    data = r.json()
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


def parse_json(text: str) -> dict:
    """Pull the JSON object out of a model reply, tolerating code fences or stray prose."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else text[text.find("{"): text.rfind("}") + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise LLMError(f"Could not parse JSON from model reply: {e}") from e
