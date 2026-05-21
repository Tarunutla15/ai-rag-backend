"""LLM service (OpenAI or Groq OpenAI-compatible API). Embeddings stay on OpenAI in embedding.py."""
from openai import OpenAI
from typing import Iterator, List, Dict, Optional
import json
import logging
import re

from app.config import settings

logger = logging.getLogger(__name__)

# Cap for RAW CODE sent to the LLM (line-boundary truncation below this size avoids mid-token cuts).
RAW_CODE_MAX_CHARS = 8000


def _truncate_raw_code(text: str, max_chars: int = RAW_CODE_MAX_CHARS) -> str:
    """Truncate at line boundaries so functions/classes are less often split mid-line."""
    if len(text) <= max_chars:
        return text
    lines = text.split("\n")
    parts: List[str] = []
    total = 0
    for line in lines:
        add_len = len(line) if not parts else len(line) + 1
        if total + add_len > max_chars:
            break
        parts.append(line)
        total += add_len
    return "\n".join(parts) + f"\n... [truncated, {len(text)} chars total]"


def _is_groq_provider() -> bool:
    return (getattr(settings, "LLM_PROVIDER", "") or "").lower().strip() == "groq"


def _trim_chat_history_for_budget(messages: List[Dict], max_chars: int) -> List[Dict]:
    """Drop oldest turns until total content length is under max_chars (keeps recent context)."""
    if max_chars <= 0 or not messages:
        return messages
    trimmed = list(messages)
    while trimmed:
        total = sum(len((m.get("content") or "")) for m in trimmed)
        if total <= max_chars:
            return trimmed
        trimmed = trimmed[1:]
    return trimmed


class LLMService:
    """Service for querying OpenAI LLM with conversation memory."""
    
    # Default system prompt for RAG chatbot - STRICT CONTEXT-ONLY MODE WITH HELPFUL GUIDANCE
    DEFAULT_SYSTEM_PROMPT = """You are a RAG assistant: answers must come ONLY from the RETRIEVED CONTEXT (PDF excerpts, RAW TABLE, RAW CODE blocks). Do not invent facts not grounded in that text.

HOW TO READ REAL PDF CONTEXT (VERY IMPORTANT):
- Chunks are messy: multi-column layouts, OCR, split tables, and broken lines are normal. Interpret charitably:
  stitch meaning across fragments, use RAW TABLE / RAW CODE even when formatting looks ugly.
- The ingest tag (e.g. Technology: python) means "this document/course is tagged as Python". It is NOT a list of
  question phrases the user is allowed to ask. Any normal Python course subtopic (modules, scope, decorators,
  loops, OOP, …) may still be answered if the excerpts OR attached raw blocks support it—even when the exact
  section heading does not appear in the chunk.
- If excerpts contain relevant keywords or examples for the question (e.g. global/local/nonlocal/def/import for
  "variable scope"; import/__main__/packages for "modules"), you MUST explain using that material. Do NOT refuse
  because the user's wording ("variable scope", "module basics") does not literally appear next to the tag "python".

WHEN TO REFUSE:
- Refuse only when NOTHING in the retrieved excerpts or raw blocks relates to the question at all (wrong subject
  or empty substance). Do not refuse because the first excerpt looks generic if other excerpts or code are on-topic.

OTHER RULES:
- Multi-topic or comparison questions: use whatever relevant passages exist; synthesize comparisons when both sides appear.
- Use chat history only to resolve pronouns and follow-ups, not to add facts absent from RETRIEVED CONTEXT.
- Do not tell the user to "ask about Python instead" when they already asked a Python subtopic covered by the tags.
- **Figures (RAW IMAGE):** Some excerpts include a `RAW IMAGE` block with `view_url=/documents/...` and a ready-made
  Markdown image line. When the user asks to **see** or **show** a figure/diagram/image and that block is in context,
  include that Markdown line verbatim in your answer so the app can display the image. Do not say the image is
  "not available" if `view_url` is present.

DO NOT HALLUCINATE: only state what the retrieved text reasonably supports.

FORMATTING: Use compact Markdown — ## for section titles, short bullet lists, no extra blank lines between sections or list items."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", base_url: Optional[str] = None):
        """
        Args:
            api_key: API key for the completion provider
            model: Chat model id (OpenAI or Groq model name)
            base_url: If set, OpenAI-compatible endpoint (e.g. Groq https://api.groq.com/openai/v1)
        """
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url.rstrip("/")
        self.client = OpenAI(**kwargs)
        self.model = model
        self.last_usage: Optional[Dict] = None
    
    def _build_rag_messages(
        self,
        query: str,
        context_chunks: List[Dict],
        chat_history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        available_technologies: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Build OpenAI-format messages for RAG completion (shared by sync + stream)."""
        if system_prompt is None:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT

        valid_chunks = [chunk for chunk in context_chunks if chunk.get("text", "").strip()]
        print(f">>> LLM: Processing {len(valid_chunks)} valid chunks (out of {len(context_chunks)} total)", flush=True)

        if not valid_chunks:
            return [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "I couldn't find any text content in the retrieved documents to answer your question.",
                },
            ]

        def _chunk_sort_key(c: Dict) -> tuple:
            score = float(c.get("score") or c.get("hybrid_score") or 0.0)
            is_figure = (c.get("chunk_type") or "").lower() == "image_caption"
            has_raw_image = bool(c.get("raw_image"))
            return (0 if is_figure and has_raw_image else 1, -score)

        q_lower = (query or "").lower()
        seeks_figure = any(
            v in q_lower
            for v in (
                "diagram", "figure", "image", "chart", "graph", "plot", "picture",
                "show the", "show me", "see the", "view the",
            )
        )
        if seeks_figure:
            valid_chunks = sorted(valid_chunks, key=_chunk_sort_key)
        else:
            valid_chunks = sorted(
                valid_chunks,
                key=lambda c: float(c.get("score") or c.get("hybrid_score") or 0.0),
                reverse=True,
            )

        max_context_chars: Optional[int] = None
        max_per_chunk: Optional[int] = None
        max_history_chars: Optional[int] = None
        if _is_groq_provider():
            max_context_chars = max(4000, int(getattr(settings, "GROQ_MAX_CONTEXT_CHARS", 16000)))
            max_per_chunk = max(500, int(getattr(settings, "GROQ_MAX_CHARS_PER_CHUNK", 3200)))
            max_history_chars = max(0, int(getattr(settings, "GROQ_MAX_CHAT_HISTORY_CHARS", 4000)))

        technologies_in_context = set()
        for chunk in valid_chunks:
            tech = chunk.get("technology", "unknown")
            if tech and tech != "unknown" and tech != "general":
                technologies_in_context.add(tech)

        def _format_raw_table(raw_table: Dict) -> str:
            if not raw_table:
                return ""
            headers = raw_table.get("headers") or []
            rows = raw_table.get("rows") or []
            limited_rows = rows[:15]
            table_lines = []
            if headers:
                table_lines.append(" | ".join([str(h) for h in headers]))
            for r in limited_rows:
                table_lines.append(" | ".join([str(c) for c in r]))
            return "RAW TABLE:\n" + "\n".join(table_lines)

        def _format_raw_code(raw_code: Dict) -> str:
            if not raw_code:
                return ""
            code_text = raw_code.get("code_text") or ""
            return "RAW CODE:\n" + _truncate_raw_code(code_text)

        def _format_raw_image(raw_image: Dict) -> str:
            if not raw_image:
                return ""
            caption = (raw_image.get("caption") or "").strip()
            path = raw_image.get("image_path") or ""
            view_url = (raw_image.get("view_url") or "").strip()
            lines = [
                "RAW IMAGE (cropped figure from the PDF):",
                f"caption={caption}",
                f"storage_path={path}",
            ]
            if view_url:
                lines.append(f"view_url={view_url}")
                alt = (caption[:120] + "…") if len(caption) > 120 else (caption or "Figure from PDF")
                lines.append("Include this exact Markdown in your reply so the user can see the image:")
                lines.append(f"![{alt}]({view_url})")
            return "\n".join(lines)

        context_parts: List[str] = []
        running = 0
        chunk_index = 0
        for chunk in valid_chunks:
            body = (chunk.get("text") or "").strip()
            if max_per_chunk and len(body) > max_per_chunk:
                body = body[:max_per_chunk] + "\n... [excerpt truncated for provider token limits]"
            chunk_index += 1
            base = (
                f"[Document excerpt {chunk_index} - Technology: {chunk.get('technology', 'general')}]:\n{body}"
            )
            extra = []
            if chunk.get("raw_table"):
                extra.append(_format_raw_table(chunk.get("raw_table")))
            if chunk.get("raw_code"):
                extra.append(_format_raw_code(chunk.get("raw_code")))
            if chunk.get("raw_image"):
                extra.append(_format_raw_image(chunk.get("raw_image")))
            if extra:
                base = base + "\n\n" + "\n\n".join([e for e in extra if e])
            piece_len = len(base) + 4
            if max_context_chars is not None and running + piece_len > max_context_chars:
                if not context_parts:
                    base = base[: max_context_chars - 200] + "\n... [truncated]"
                    context_parts.append(base)
                    logger.warning(
                        "LLM: Groq context budget exceeded on first chunk; hard-truncated (raise GROQ_MAX_CONTEXT_CHARS if needed)"
                    )
                else:
                    logger.warning(
                        "LLM: stopping at %s excerpt(s) for Groq context cap (~%s chars)",
                        len(context_parts),
                        max_context_chars,
                    )
                break
            context_parts.append(base)
            running += piece_len

        context = "\n\n".join(context_parts)
        if max_context_chars is not None and len(context_parts) < len(valid_chunks):
            context += "\n\n[Note: additional retrieved excerpts were omitted to fit the model provider token limits.]"

        print(f">>> LLM: Total context length: {len(context)} characters", flush=True)
        print(f">>> LLM: Technologies in context: {technologies_in_context}", flush=True)

        messages = [{"role": "system", "content": system_prompt}]
        history_src = chat_history or []
        if max_history_chars is not None and history_src:
            history_src = _trim_chat_history_for_budget(list(history_src), max_history_chars)
        if history_src:
            print(f">>> LLM: Including {len(history_src)} messages from chat history", flush=True)
            for msg in history_src:
                if msg["role"] == "user" and msg["content"].strip() == query.strip():
                    continue
                messages.append({"role": msg["role"], "content": msg["content"]})

        tech_list = ", ".join(sorted(technologies_in_context)) if technologies_in_context else "general/unknown"
        library_labels_text = ""
        if available_technologies and len(available_technologies) > 0:
            library_labels_text = f"""
OTHER DOCUMENT LABELS IN YOUR LIBRARY (orientation only — not a checklist of allowed question phrases):
{', '.join(available_technologies)}
"""

        user_prompt = f"""Answer using ONLY the RETRIEVED CONTEXT below (including RAW TABLE and RAW CODE sections).

RETRIEVED CONTEXT:
{context}

Coarse document/course tags on chunks (not a list of allowed questions): {tech_list}
{library_labels_text}
QUESTION: {query}

Instructions:
1. Read ALL excerpts. Noisy or fragmented PDF text still counts—extract whatever clearly relates to the question.
2. If any excerpt or raw block discusses the topic, answer with compact Markdown (## sections, tight bullets, no extra blank lines).
3. Do NOT say "context does not contain" when the tag matches the course and snippets relate to the question.
4. Say you cannot answer from the documents ONLY if nothing in the retrieved text bears on the question.
5. Do not add facts from training data that are not supported by the excerpts.

Write the best compact answer the retrieved text allows."""

        messages.append({"role": "user", "content": user_prompt})
        print(f">>> LLM: Total messages being sent to LLM: {len(messages)}", flush=True)
        return messages

    def _completion_max_tokens(self) -> int:
        completion_max = 900
        if _is_groq_provider():
            completion_max = min(completion_max, 768)
        return completion_max

    def _capture_usage(self, response) -> None:
        try:
            usage = getattr(response, "usage", None)
            if usage:
                self.last_usage = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            else:
                self.last_usage = None
        except Exception:
            self.last_usage = None

    def generate_response(
        self,
        query: str,
        context_chunks: List[Dict],
        chat_history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        available_technologies: Optional[List[str]] = None,
    ) -> str:
        """Generate response using LLM with context from vector search and chat history."""
        messages = self._build_rag_messages(
            query, context_chunks, chat_history, system_prompt, available_technologies
        )
        if len(messages) == 2 and messages[1].get("role") == "user" and "couldn't find any text" in (messages[1].get("content") or ""):
            return messages[1]["content"]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=self._completion_max_tokens(),
        )
        self._capture_usage(response)
        return (response.choices[0].message.content or "").strip()

    def stream_response(
        self,
        query: str,
        context_chunks: List[Dict],
        chat_history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        available_technologies: Optional[List[str]] = None,
    ) -> Iterator[str]:
        """Yield completion text deltas for SSE streaming."""
        messages = self._build_rag_messages(
            query, context_chunks, chat_history, system_prompt, available_technologies
        )
        if len(messages) == 2 and messages[1].get("role") == "user" and "couldn't find any text" in (messages[1].get("content") or ""):
            yield messages[1]["content"]
            self.last_usage = None
            return

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=self._completion_max_tokens(),
            stream=True,
        )
        self.last_usage = None
        for chunk in stream:
            if getattr(chunk, "usage", None):
                self._capture_usage(chunk)
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue
            delta = getattr(choice.delta, "content", None) if choice.delta else None
            if delta:
                yield delta
    
    def generate_response_simple(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Generate response with raw messages (for custom use cases).
        
        Args:
            messages: List of messages in OpenAI format
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            LLM generated response
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content.strip()

    def is_context_sufficient(
        self,
        query: str,
        context_chunks: List[Dict],
        max_chunks: int = 5,
        max_chars_per_chunk: int = 400
    ) -> bool:
        """
        Check if retrieved context is sufficient to answer the query.
        Returns True/False based on LLM judgment.
        """
        if not query or not query.strip():
            return False
        if not context_chunks:
            return False

        # Heuristic: if question asks for examples/code/syntax, require example-like text in context
        q = query.lower()
        wants_examples = any(
            kw in q for kw in (
                "example", "examples", "sample", "code", "syntax", "implementation", "demo"
            )
        )
        if wants_examples:
            example_markers = ("example", "for example", "e.g.", "eg.", "sample", "code", "syntax")
            has_example = False
            for c in context_chunks[:max_chunks]:
                t = (c.get("text") or "").lower()
                if any(m in t for m in example_markers):
                    has_example = True
                    break
            if not has_example:
                return False

        # Build a compact context excerpt for evaluation
        excerpts = []
        for c in context_chunks[:max_chunks]:
            text = (c.get("text") or "").strip()
            if not text:
                continue
            excerpts.append(text[:max_chars_per_chunk])

        if not excerpts:
            return False

        context = "\n\n".join(f"[Excerpt {i+1}] {t}" for i, t in enumerate(excerpts))

        prompt = f"""You evaluate PDF snippets for a tutor chatbot. Text may be OCR-noisy or fragmented.

Answer "yes" if ANY excerpt could help answer the question even partially (same course/subject, related terms,
code, or tables). Answer "no" only if excerpts are clearly off-topic or empty of usable content.

Question: {query.strip()}

Retrieved excerpts:
{context}

Answer only "yes" or "no"."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=3
        )

        text = (response.choices[0].message.content or "").strip().lower()
        return text.startswith("y")

    def rewrite_query(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Rewrite a user query into a better retrieval query.
        Returns the rewritten query string; falls back to original on failure.
        """
        if not query or not query.strip():
            return query

        history_block = ""
        if chat_history:
            # Use the last few messages for context (short)
            recent = chat_history[-4:]
            history_lines = []
            for m in recent:
                role = m.get("role", "user")
                content = (m.get("content") or "").strip()
                if content:
                    history_lines.append(f"{role}: {content}")
            if history_lines:
                history_block = "\nRecent conversation:\n" + "\n".join(history_lines) + "\n"

        prompt = f"""Rewrite the user question into a concise, retrieval-optimized query.
Keep the same intent, add missing specifics and likely keywords.
Do not answer the question.
Return JSON with a single key: "rewrite_query".

{history_block}User question: {query.strip()}

Example output:
{{"rewrite_query": "Explain Java HashMap internal working with examples"}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=120
        )

        text = (response.choices[0].message.content or "").strip()
        if not text:
            return query

        # Try to parse JSON if present
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                rewritten = (data.get("rewrite_query") or "").strip()
                return rewritten if rewritten else query
        except Exception:
            pass

        # Fallback: return raw text (first line)
        first_line = text.splitlines()[0].strip()
        return first_line if first_line else query
