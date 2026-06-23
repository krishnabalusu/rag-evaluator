import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any
from config.settings import (
    LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL,
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE
)

SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer ONLY based on the provided context.
If the context does not contain enough information, say "I don't have enough information to answer this question."
Never make up facts. Never use knowledge outside the provided context.
Always cite the source at the end of your answer."""


def _build_prompt(query: str, context: str) -> str:
    return f"""Context:
{context}

Question: {query}

Answer based only on the context above:"""


def generate_answer(query: str, context: str) -> Dict[str, Any]:
    """Send query + context to LLM and return grounded answer."""

    if LLM_PROVIDER == "openai":
        return _generate_openai(query, context)
    elif LLM_PROVIDER == "anthropic":
        return _generate_anthropic(query, context)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. Use 'openai' or 'anthropic'.")


def _generate_openai(query: str, context: str) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(query, context)},
        ],
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
    )

    answer = response.choices[0].message.content.strip()
    tokens_used = response.usage.total_tokens

    return {
        "answer": answer,
        "model": OPENAI_MODEL,
        "tokens_used": tokens_used,
        "provider": "openai",
    }


def _generate_anthropic(query: str, context: str) -> Dict[str, Any]:
    try:
        import anthropic
    except ImportError:
        raise ImportError("Run: pip install anthropic")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": _build_prompt(query, context)},
        ],
    )

    answer = response.content[0].text.strip()
    tokens_used = response.usage.input_tokens + response.usage.output_tokens

    return {
        "answer": answer,
        "model": ANTHROPIC_MODEL,
        "tokens_used": tokens_used,
        "provider": "anthropic",
    }