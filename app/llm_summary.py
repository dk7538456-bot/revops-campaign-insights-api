"""Optional OpenAI-compatible chat completion summary adapter."""

from __future__ import annotations

import os
from typing import Any

import httpx


async def summarize_analysis(analysis: dict[str, Any]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")
    if not api_key or not model:
        raise RuntimeError("Set OPENAI_API_KEY and OPENAI_MODEL to enable summaries")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "Summarize these aggregate marketing metrics in 3 concise bullets. State observations only; do not invent causes or promise results."},
            {"role": "user", "content": str({"totals": analysis["totals"], "kpis": analysis["kpis"], "suggestions": analysis["suggestions"]})},
        ],
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()
