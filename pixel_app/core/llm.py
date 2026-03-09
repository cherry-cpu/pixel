from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any


SYSTEM_PROMPT = """You are a helpful assistant that converts a user's natural language photo search query
into STRICT JSON for a photo library.

Return ONLY valid JSON with this exact schema (no markdown, no explanation):
{"people": [], "tags": [], "text": "", "date_from": null, "date_to": null, "limit": 50}

Schema:
- people: array of person names (e.g. ["Alice", "Bob"]), empty [] if none
- tags: array of tags/keywords (e.g. ["beach", "sunset"]), empty [] if none
- text: string for captions/filenames, "" if none
- date_from: "YYYY-MM-DD" or null
- date_to: "YYYY-MM-DD" or null
- limit: number between 1 and 200

Guidelines:
- If user says "me", put "me" in tags, not in people.
- Extract person names into people when clearly a name.
- Put vague terms (beach, sunset, party) in tags.
"""


@dataclass(frozen=True)
class ParsedQuery:
    people: list[str]
    tags: list[str]
    text: str
    date_from: str | None
    date_to: str | None
    limit: int


def _extract_json(content: str) -> dict[str, Any] | None:
    """Extract JSON from LLM response (may be wrapped in markdown code blocks)."""
    if not content or not isinstance(content, str):
        return None
    content = content.strip()
    # Try markdown code block first: ```json ... ``` or ``` ... ```
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try raw JSON or first {...} object
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    obj_match = re.search(r"\{[\s\S]*\}", content)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _coerce(obj: dict[str, Any]) -> ParsedQuery:
    people = [str(x).strip() for x in (obj.get("people") or []) if str(x).strip()]
    tags = [str(x).strip() for x in (obj.get("tags") or []) if str(x).strip()]
    text = str(obj.get("text") or "").strip()
    date_from = obj.get("date_from")
    date_to = obj.get("date_to")
    limit = int(obj.get("limit") or 50)
    limit = max(1, min(200, limit))
    return ParsedQuery(
        people=people,
        tags=tags,
        text=text,
        date_from=str(date_from) if date_from else None,
        date_to=str(date_to) if date_to else None,
        limit=limit,
    )


def parse_query_with_llm(user_query: str) -> ParsedQuery | None:
    """
    Returns ParsedQuery if a provider is configured, else None.
    Prefers Groq, falls back to Hugging Face Inference.
    """
    user_query = (user_query or "").strip()
    if not user_query:
        return None

    groq_key = os.getenv("GROQ_API_KEY")
    hf_token = os.getenv("HF_TOKEN")

    if groq_key:
        try:
            from groq import Groq

            client = Groq(api_key=groq_key)
            resp = client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_query},
                ],
                temperature=0.1,
                max_tokens=300,
            )
            content = (resp.choices[0].message.content or "").strip()
            obj = _extract_json(content)
            if obj is not None:
                return _coerce(obj)
        except Exception:
            pass

    if hf_token:
        try:
            from huggingface_hub import InferenceClient

            model = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")
            client = InferenceClient(model=model, token=hf_token)
            resp = client.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_query},
                ],
                max_tokens=300,
                temperature=0.1,
            )
            msg = resp.choices[0].message
            content = (getattr(msg, "content", None) or "").strip()
            obj = _extract_json(content)
            if obj is not None:
                return _coerce(obj)
        except Exception:
            pass

    return None

