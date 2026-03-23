from typing import List, Dict, Any

from config import (
    LLM_PROVIDER,
    MAIN_MODEL,
    CHATBOT_MODEL,
    GOOGLE_AI_API_KEY,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
)

# Lazy singletons
_openai_client = None
_anthropic_client = None
_gemini_configured = False


def _ensure_gemini_configured():
    """
    Configure google.generativeai once per process.

    Requires: pip install google-generativeai
    """
    global _gemini_configured
    if _gemini_configured:
        return

    if not GOOGLE_AI_API_KEY:
        raise RuntimeError("GOOGLE_AI_API_KEY is not set")

    import google.generativeai as genai  # type: ignore
    genai.configure(api_key=GOOGLE_AI_API_KEY)
    _gemini_configured = True


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI  # type: ignore
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic  # type: ignore
        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


def generate_text(system_prompt: str, user_content: str, model: str = None, max_tokens: int = 1500) -> str:
    """
    Unified text-generation (for extraction, scoring, detailed analysis).
    """
    model = model or MAIN_MODEL

    # ---------------- Google Gemini (google-generativeai) ----------------
    if LLM_PROVIDER == "google":
        _ensure_gemini_configured()
        import google.generativeai as genai  # type: ignore

        full_prompt = system_prompt + "\n\n" + user_content
        gm = genai.GenerativeModel(model)
        resp = gm.generate_content(
            full_prompt,
            generation_config={"max_output_tokens": max_tokens, "temperature": 0.2},
        )
        return getattr(resp, "text", "") or ""

    # ---------------- OpenAI ----------------
    if LLM_PROVIDER == "openai":
        client = _get_openai_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""

    # ---------------- Anthropic (Claude) ----------------
    if LLM_PROVIDER == "anthropic":
        client = _get_anthropic_client()
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        parts: List[str] = []
        for block in msg.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(block["text"])
        return "\n".join(parts)

    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def chat(system_prompt: str, messages: List[Dict[str, str]], model: str = None, max_tokens: int = 1500) -> str:
    """
    Unified chat interface (used by AI Chatbot tab).

    messages: [{"role": "user"/"assistant", "content": "..."}]
    """
    model = model or CHATBOT_MODEL

    # ---------------- Google Gemini (google-generativeai) ----------------
    if LLM_PROVIDER == "google":
        _ensure_gemini_configured()
        import google.generativeai as genai  # type: ignore

        gm = genai.GenerativeModel(model)

        # Flatten messages into a single prompt, keeping system prompt separate.
        prompt_parts: List[str] = []
        if system_prompt:
            prompt_parts.append(f"[SYSTEM]\n{system_prompt}")
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            prompt_parts.append(f"[{role.upper()}]\n{content}")

        full_prompt = "\n\n".join(prompt_parts)
        resp = gm.generate_content(
            full_prompt,
            generation_config={"max_output_tokens": max_tokens, "temperature": 0.3},
        )
        return getattr(resp, "text", "") or ""

    # ---------------- OpenAI ----------------
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

    # ---------------- Anthropic (Claude) ----------------
    if LLM_PROVIDER == "anthropic":
        client = _get_anthropic_client()
        anthro_msgs = []
        if system_prompt:
            anthro_msgs.append({"role": "user", "content": system_prompt})
        anthro_msgs.extend(messages)
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=anthro_msgs,
        )
        parts: List[str] = []
        for block in msg.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(block["text"])
        return "\n".join(parts)

    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")
