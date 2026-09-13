"""
Gemini LLM Integration
========================
Wraps the Google Generative AI client for use in the RAG pipeline.

🧠 AI CONCEPT — LLM in RAG:
    The LLM's job in RAG is NOT to recall facts from training data.
    Instead, we give it retrieved chunks as context and instruct it:
    "Answer ONLY based on the provided context. Cite your sources."
    
    This approach solves a key LLM problem: HALLUCINATION.
    Without RAG, the LLM might confidently make up facts that weren't
    in your documents. With RAG, it can only use what we retrieved.
    
    The LLM's actual job here:
      1. Synthesize multiple retrieved chunks into a coherent answer
      2. Rephrase technical content clearly
      3. Identify when the context doesn't contain the answer
         (and say "I don't know" instead of hallucinating)

🧠 AI CONCEPT — Prompt Engineering:
    The system prompt is a set of instructions that shapes LLM behavior.
    We use a technique called "grounded generation" with explicit rules:
      - Always cite which document your answer comes from
      - Don't answer from general knowledge
      - Indicate confidence level
      - Be concise but complete
"""
import logging
from typing import Optional, Generator
import httpx
import google.generativeai as genai
from app.core.config import get_settings

logger = logging.getLogger(__name__)

_model = None


def get_llm():
    """Initialize Gemini client (once, cached)."""
    global _model
    if _model is None:
        settings = get_settings()
        provider = _resolve_provider(settings)
        _model = _build_model(provider, settings)
        logger.info("LLM initialized via %s", provider)
    return _model


def _build_model(provider: str, settings):
    if provider == "mistral":
        if not settings.mistral_api_key:
            raise RuntimeError("MISTRAL_API_KEY is not configured")
        return {
            "provider": "mistral",
            "api_key": settings.mistral_api_key,
            "model": settings.mistral_model,
            "system_prompt": _build_system_prompt(),
        }

    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    genai.configure(api_key=settings.gemini_api_key)
    return genai.GenerativeModel(
        model_name=settings.gemini_model,
        generation_config=genai.GenerationConfig(
            temperature=0.2,       # Low temp = more factual, less creative
            top_p=0.9,
            max_output_tokens=2048,
        ),
        system_instruction=_build_system_prompt(),
    )


def _resolve_provider(settings) -> str:
    provider = (settings.llm_provider or "auto").strip().lower()

    if provider in {"gemini", "mistral"}:
        return provider

    if settings.mistral_api_key:
        return "mistral"
    if settings.gemini_api_key:
        return "gemini"

    raise RuntimeError("Set either MISTRAL_API_KEY or GEMINI_API_KEY")


def _build_system_prompt() -> str:
    return """You are an AI Research Assistant. Your job is to answer questions 
based STRICTLY on the provided document context. Follow these rules:

1. GROUNDED ANSWERS: Only use information from the provided context.
   Never use your general knowledge to fill gaps.

2. CITATIONS: Always cite your sources using [Source: filename, chunk X].
   Example: "According to the paper [Source: attention_paper.pdf, chunk 3], ..."

3. ACKNOWLEDGE GAPS: If the context doesn't contain enough information
   to answer the question, say: "The provided documents do not contain 
   sufficient information to answer this question."

4. MULTI-TURN: You will receive conversation history. Use it to understand
   follow-up questions and maintain coherent conversation.

5. FORMAT: Use clear, readable formatting. Use bullet points and headers
   where appropriate. Be concise but thorough.

6. ACCURACY: Do not speculate or extrapolate beyond what the documents say."""


def build_rag_prompt(
    query: str,
    context_chunks: list[dict],
    conversation_history: list[dict],
) -> str:
    """
    🧠 AI CONCEPT — Context Window Construction:
    
    We build the prompt by concatenating:
    1. System prompt (already set on model initialization)
    2. Conversation history (for multi-turn memory)
    3. Retrieved context chunks (the "retrieval" in RAG)
    4. The user's current question
    
    The order matters. Context comes BEFORE the question so the
    model reads the relevant info first, then sees what's being asked.
    """
    # Format conversation history
    history_text = ""
    if conversation_history:
        history_lines = []
        for msg in conversation_history[-6:]:   # Last 3 turns (6 messages)
            role = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {msg['content']}")
        history_text = "\n".join(history_lines)

    # Format retrieved context
    context_text = ""
    for i, chunk in enumerate(context_chunks, start=1):
        meta = chunk.get("metadata", {})
        filename = meta.get("filename", "unknown")
        chunk_idx = meta.get("chunk_index", i)
        page = meta.get("page_number", -1)
        page_info = f", page {page}" if page > 0 else ""
        
        context_text += (
            f"--- [Source {i}: {filename}, chunk {chunk_idx}{page_info}] ---\n"
            f"{chunk['content']}\n\n"
        )

    prompt_parts = []

    if history_text:
        prompt_parts.append(f"=== Conversation History ===\n{history_text}\n")

    prompt_parts.append(f"=== Retrieved Context ===\n{context_text}")
    prompt_parts.append(f"=== Current Question ===\n{query}")
    prompt_parts.append(
        "\nAnswer the question based on the context above. "
        "Include citations in your response."
    )

    return "\n".join(prompt_parts)


def generate_answer(
    query: str,
    context_chunks: list[dict],
    conversation_history: list[dict],
) -> tuple[str, int]:
    """
    Generate an answer using Gemini given retrieved context.
    Returns (answer_text, tokens_used).
    """
    model = get_llm()
    prompt = build_rag_prompt(query, context_chunks, conversation_history)
    settings = get_settings()

    try:
        if isinstance(model, dict) and model.get("provider") == "mistral":
            answer, tokens = _generate_with_mistral(model, prompt)
        else:
            response = model.generate_content(prompt)
            answer = response.text if response.text else "I was unable to generate a response."
            tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens = getattr(response.usage_metadata, "total_token_count", 0)
    except Exception as e:
        if _is_quota_error(e):
            alternate_provider = "mistral" if _resolve_provider(settings) == "gemini" and settings.mistral_api_key else "gemini"
            if alternate_provider == "mistral" and settings.mistral_api_key:
                logger.warning("Gemini quota exhausted; retrying with Mistral: %s", e)
                answer, tokens = _generate_with_mistral(
                    {
                        "provider": "mistral",
                        "api_key": settings.mistral_api_key,
                        "model": settings.mistral_model,
                        "system_prompt": _build_system_prompt(),
                    },
                    prompt,
                )
            else:
                logger.warning("LLM quota exhausted; using retrieval fallback: %s", e)
                return _build_retrieval_fallback(query, context_chunks), 0
        else:
            raise

    return answer, tokens


def _generate_with_mistral(model: dict, prompt: str) -> tuple[str, int]:
    headers = {
        "Authorization": f"Bearer {model['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model["model"],
        "messages": [
            {"role": "system", "content": model["system_prompt"]},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 2048,
    }

    response = httpx.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices", [])
    message = choices[0].get("message", {}) if choices else {}
    answer = message.get("content") or "I was unable to generate a response."
    tokens = data.get("usage", {}).get("total_tokens", 0)
    return answer, tokens


def _is_quota_error(error: Exception) -> bool:
    message = str(error).lower()
    error_type = type(error).__name__.lower()
    return (
        "quota" in message
        or "rate limit" in message
        or "429" in message
        or "resourceexhausted" in error_type
        or "toomanyrequests" in error_type
    )


def _build_retrieval_fallback(query: str, context_chunks: list[dict]) -> str:
    if not context_chunks:
        return (
            "The Gemini API is currently unavailable due to quota limits, and no "
            "retrieved context was available to summarize. Please upload documents "
            "or try again after the quota window resets."
        )

    lines = [
        "The Gemini API quota is currently exhausted, so I could not generate a synthesized answer.",
        "Based on the retrieved documents, the most relevant excerpts are:",
    ]

    for chunk in context_chunks[:3]:
        meta = chunk.get("metadata", {})
        filename = meta.get("filename", "unknown")
        chunk_idx = meta.get("chunk_index", "?")
        preview = " ".join(chunk.get("content", "").split())[:320]
        lines.append(f"- {filename} (chunk {chunk_idx}): {preview}")

    lines.append(
        "If you want a synthesized answer from Gemini, set up billing or use a project/model with available quota."
    )

    return "\n".join(lines)
