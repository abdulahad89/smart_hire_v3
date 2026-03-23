from typing import List, Dict, Any

from config import (
    LLM_PROVIDER,
    MAIN_MODEL,
    CHATBOT_MODEL,
    GOOGLE_AI_API_KEY,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
)

_gemini_client = None
_openai_client = None
_anthropic_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai  # pip install google-genai
        _gemini_client = genai.Client(api_key=GOOGLE_AI_API_KEY)
    return _gemini_client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI  # pip install openai
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic  # pip install anthropic
        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


def generate_text(system_prompt: str, user_content: str, model: str = None, max_tokens: int = 1500) -> str:
    model = model or MAIN_MODEL

    if LLM_PROVIDER == "google":
        client = _get_gemini_client()
        from google.genai import types  # type: ignore
        resp = client.models.generate_content(
            model=model,
            contents=[{"role": "user", "parts": [system_prompt + "\n\n" + user_content]}],
            config=types.GenerateContentConfig(max_output_tokens=max_tokens, temperature=0.2),
        )
        return getattr(resp, "text", "")

    if LLM_PROVIDER == "openai":
        client = _get_openai_client()
        resp = client.chat.completions.create(  # Chat Completions API[web:23][web:26]
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""

    if LLM_PROVIDER == "anthropic":
        client = _get_anthropic_client()
        msg = client.messages.create(  # Claude Messages API[web:24][web:30][web:33]
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        parts = []
        for block in msg.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(block["text"])
        return "\n".join(parts)

    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def chat(system_prompt: str, messages: List[Dict[str, str]], model: str = None, max_tokens: int = 1500) -> str:
    model = model or CHATBOT_MODEL

    if LLM_PROVIDER == "google":
        client = _get_gemini_client()
        from google.genai import types  # type: ignore

        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [system_prompt]})
        for m in messages:
            contents.append({"role": m["role"], "parts": [m["content"]]})

        resp = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(max_output_tokens=max_tokens, temperature=0.3),
        )
        return getattr(resp, "text", "")

    if LLM_PROVIDER == "openai":
        client = _get_openai_client()
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        resp = client.chat.completions.create(
            model=model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""

    if LLM_PROVIDER == "anthropic":
        client = _get_anthropic_client()
        anthro_messages = []
        if system_prompt:
            anthro_messages.append({"role": "user", "content": system_prompt})
        anthro_messages.extend(messages)
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=anthro_messages,
        )
        parts = []
        for block in msg.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(block["text"])
        return "\n".join(parts)

    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")
