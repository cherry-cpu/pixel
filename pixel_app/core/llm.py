from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


SYSTEM_PROMPT = """You are a helpful assistant that converts a user's natural language photo search query
into STRICT JSON for a photo library.

Return ONLY valid JSON with this schema:
{
  "people": [string],        // person names, empty if none
  "tags": [string],          // tags/keywords, empty if none
  "text": string,            // free text to match captions/filenames, "" if none
  "date_from": string|null,  // ISO date "YYYY-MM-DD" if specified, else null
  "date_to": string|null,    // ISO date "YYYY-MM-DD" if specified, else null
  "limit": number            // 1..200
}

Guidelines:
- If user mentions "me", do not guess a name; put it in tags as "me".
- Prefer extracting person names into people.
- If unsure, put content into tags/text rather than inventing fields.
"""


@dataclass(frozen=True)
class ParsedQuery:
    people: list[str]
    tags: list[str]
    text: str
    date_from: str | None
    date_to: str | None
    limit: int


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
            content = resp.choices[0].message.content
            obj = json.loads(content)
            return _coerce(obj)
        except Exception:
            # Fall through to HF or keyword search
            pass

    if hf_token:
        try:
            from huggingface_hub import InferenceClient

            model = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
            client = InferenceClient(model=model, token=hf_token)
            text = client.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_query},
                ],
                max_tokens=300,
                temperature=0.1,
            ).choices[0].message.content
            obj = json.loads(text)
            return _coerce(obj)
        except Exception:
            pass

    return None

