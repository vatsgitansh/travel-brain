"""
chat.py — RAG-powered chat endpoint for Travel Brain.

Supports two LLM providers (auto-detected from .env):
  - Gemini (free via Google AI Studio — recommended)
  - OpenAI GPT-4o-mini (paid, ~$0.0002 per query)

Flow:
  1. Embed the user's query (local model, free)
  2. Retrieve top-k relevant travel chunks from ChromaDB
  3. Build a RAG prompt: Travel Brain persona + retrieved context + user question
  4. Stream the LLM response via SSE
"""

import json
import logging
import os
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from travel_brain import config
from travel_brain.processing.embedder import embed_texts
from travel_brain.vectordb.chroma_client import ChromaClient

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str    # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=15)

class ChatResponse(BaseModel):
    answer: str
    context_chunks_used: int
    provider: str


# ── DB singleton ──────────────────────────────────────────────────────────────

_db: Optional[ChromaClient] = None

def get_db() -> ChromaClient:
    global _db
    if _db is None:
        _db = ChromaClient()
    return _db


# ── RAG: Retrieve context ─────────────────────────────────────────────────────

def retrieve_context(query: str, top_k: int = 5) -> list[dict]:
    """Embed the query and pull the most relevant travel chunks from all namespaces."""
    try:
        query_vec = embed_texts([query])[0]
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return []

    db = get_db()
    all_results: list[dict] = []

    for location in config.LOCATIONS:
        try:
            results = db.query(
                query_embedding=query_vec,
                top_k=top_k,
                namespace=location,
                filter=None,
            )
            all_results.extend(results)
        except Exception:
            pass  # Empty namespace — skip silently

    # Deduplicate by first 100 chars of text, sort by score
    seen: set[str] = set()
    unique: list[dict] = []
    for r in sorted(all_results, key=lambda x: x.get("score", 0), reverse=True):
        key = r.get("text", "")[:100]
        if key and key not in seen:
            seen.add(key)
            unique.append(r)

    return unique[:top_k]


# ── RAG: Build prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Travel Brain — a knowledgeable and enthusiastic AI travel companion specialising in Bali and Dubai.

Your knowledge comes from thousands of real traveller experiences: blog posts, Reddit discussions, and YouTube vlogs. You give honest, specific, practical advice.

Guidelines:
- Be conversational and warm, like a well-travelled friend giving advice
- Give specific names, prices, and locations whenever possible
- Mention costs in local currency and USD when useful
- Proactively mention safety concerns or common scams when relevant
- Use clear formatting (bullet points, headers) for longer answers
- If asked about something outside your knowledge, be honest about it
- Never fabricate specific names, prices, or addresses you are not sure about"""


def build_messages_openai(query: str, context_chunks: list[dict], history: list[ChatMessage]) -> list[dict]:
    """Build the OpenAI-format messages list."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if context_chunks:
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            meta = chunk.get("metadata", {})
            text = chunk.get("text", "").strip()
            if text:
                context_parts.append(f"[Reference {i}]\n{text}")
        context_block = "\n\n---\n\n".join(context_parts)
        messages.append({
            "role": "system",
            "content": (
                "Use the following travel knowledge to answer the user's question. "
                "Synthesise it into a natural, helpful response — don't just list it:\n\n"
                + context_block
            )
        })
    else:
        messages.append({
            "role": "system",
            "content": "No specific knowledge was found in the database for this query. "
                       "Use your general knowledge and be transparent that you don't have "
                       "specific sourced information on this topic."
        })

    for msg in history[-10:]:
        if msg.role in ("user", "assistant"):
            messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": query})
    return messages


def build_gemini_prompt(query: str, context_chunks: list[dict], history: list[ChatMessage]) -> str:
    """Build a single prompt string for Gemini (handles its own conversation internally)."""
    parts = [SYSTEM_PROMPT, "\n\n"]

    if context_chunks:
        parts.append("TRAVEL KNOWLEDGE BASE:\n")
        for i, chunk in enumerate(context_chunks, 1):
            text = chunk.get("text", "").strip()
            if text:
                parts.append(f"[{i}] {text}\n\n")
        parts.append(
            "Use the above knowledge to answer the user's question naturally. "
            "Synthesise the information — don't just repeat it verbatim.\n\n"
        )

    if history:
        parts.append("CONVERSATION SO FAR:\n")
        for msg in history[-10:]:
            label = "User" if msg.role == "user" else "Travel Brain"
            parts.append(f"{label}: {msg.content}\n")
        parts.append("\n")

    parts.append(f"User: {query}\nTravel Brain:")
    return "".join(parts)


# ── LLM streaming ─────────────────────────────────────────────────────────────

async def stream_gemini(query: str, context_chunks: list[dict], history: list[ChatMessage]) -> AsyncGenerator[str, None]:
    """Stream response from Gemini 2.0 Flash Lite (free tier) using the google-genai SDK."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=config.GEMINI_API_KEY)

        # Build conversation history for Gemini/Gemma
        gemini_history = []

        # First message: system persona + travel knowledge context
        context_parts_text = []
        if context_chunks:
            for i, chunk in enumerate(context_chunks, 1):
                text = chunk.get("text", "").strip()
                if text:
                    context_parts_text.append(f"[{i}] {text}")

        setup_msg = SYSTEM_PROMPT
        if context_parts_text:
            setup_msg += (
                "\n\nRelevant travel knowledge for this conversation:\n\n"
                + "\n\n".join(context_parts_text)
                + "\n\nUse this to give accurate, specific answers."
            )

        gemini_history.append(types.Content(role="user",  parts=[types.Part(text=setup_msg)]))
        gemini_history.append(types.Content(role="model", parts=[types.Part(text="Understood! I'm Travel Brain, ready to give expert travel advice about Bali and Dubai using the knowledge provided.")]))

        # Add conversation history
        for msg in history[-10:]:
            role = "user" if msg.role == "user" else "model"
            gemini_history.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))

        config_gen = types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=1500,
        )

        async for chunk in await client.aio.models.generate_content_stream(
            model="gemma-3-27b-it",
            contents=gemini_history + [types.Content(role="user", parts=[types.Part(text=query)])],
            config=config_gen,
        ):
            if chunk.text:
                event = json.dumps({"type": "delta", "content": chunk.text})
                yield f"data: {event}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        logger.error(f"Gemini error: {e}")
        error_event = json.dumps({"type": "error", "content": f"Gemini error: {str(e)}"})
        yield f"data: {error_event}\n\n"


async def stream_openai(query: str, context_chunks: list[dict], history: list[ChatMessage]) -> AsyncGenerator[str, None]:
    """Stream response from OpenAI GPT-4o-mini."""
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        messages = build_messages_openai(query, context_chunks, history)

        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=1500,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                event = json.dumps({"type": "delta", "content": delta.content})
                yield f"data: {event}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        error_event = json.dumps({"type": "error", "content": f"OpenAI error: {str(e)}"})
        yield f"data: {error_event}\n\n"


async def stream_no_llm(context_chunks: list[dict]) -> AsyncGenerator[str, None]:
    """
    Fallback when no LLM key is configured.
    Returns retrieved content formatted as a clean answer.
    """
    if not context_chunks:
        content = (
            "I don't have specific information about that in my knowledge base yet. "
            "Try running the pipeline to add more travel content, or ask about Bali or Dubai specifically.\n\n"
            "💡 **To get full AI answers:** Add your `GEMINI_API_KEY` to the `.env` file. "
            "Get a free key at [aistudio.google.com](https://aistudio.google.com/apikey)."
        )
    else:
        lines = ["Here's what I found in the knowledge base:\n"]
        for i, chunk in enumerate(context_chunks[:3], 1):
            meta  = chunk.get("metadata", {})
            text  = chunk.get("text", "")[:400].strip()
            title = meta.get("source_title", "Travel info")
            lines.append(f"**{i}. {title}**\n{text}\n")
        lines.append(
            "\n---\n💡 *Add your `GEMINI_API_KEY` (free at [aistudio.google.com](https://aistudio.google.com/apikey)) "
            "for full conversational AI answers.*"
        )
        content = "\n".join(lines)

    yield f"data: {json.dumps({'type': 'delta', 'content': content})}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# ── Endpoint: POST /chat/stream ───────────────────────────────────────────────

@router.post(
    "/stream",
    summary="Stream a RAG-powered chat response",
    description="""
Send a message. The system will:
1. Embed your query and search the travel knowledge base
2. Feed the relevant context to the LLM (Gemini or OpenAI)
3. Stream back a natural language answer

**SSE event types:**
- `{"type": "delta", "content": "..."}` — streamed tokens
- `{"type": "done"}` — stream finished
- `{"type": "error", "content": "..."}` — on failure

**Supported LLM providers** (auto-detected from `.env`):
- `GEMINI_API_KEY` → uses Gemini 2.0 Flash (free tier)
- `OPENAI_API_KEY` → uses GPT-4o-mini
    """,
)
async def stream_chat(req: ChatRequest) -> StreamingResponse:
    context_chunks = retrieve_context(req.message, top_k=req.top_k)
    logger.info(
        f"[{config.LLM_PROVIDER or 'no-llm'}] "
        f"Retrieved {len(context_chunks)} chunks for: {req.message[:60]}"
    )

    if config.LLM_PROVIDER == "gemini":
        gen = stream_gemini(req.message, context_chunks, req.history)
    elif config.LLM_PROVIDER == "openai":
        gen = stream_openai(req.message, context_chunks, req.history)
    else:
        gen = stream_no_llm(context_chunks)

    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Endpoint: GET /chat/provider ──────────────────────────────────────────────

@router.get("/provider", summary="Check which LLM provider is active")
async def get_provider() -> dict:
    return {
        "provider": config.LLM_PROVIDER or "none",
        "gemini_configured": bool(config.GEMINI_API_KEY),
        "openai_configured": bool(config.OPENAI_API_KEY),
        "message": (
            f"Using {config.LLM_PROVIDER}"
            if config.LLM_PROVIDER
            else "No LLM configured — add GEMINI_API_KEY to .env for full AI responses"
        ),
    }
