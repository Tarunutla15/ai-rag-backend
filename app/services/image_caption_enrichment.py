"""
Vision captions for extracted PDF figures (Groq / Llama 4 Scout by default).

Appends a grounded visual description to each image block before chunking/embed,
so retrieval is not limited to same-page text or generic "figure on page N" boilerplate.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)

# Groq: base64 request body max 4MB; keep raw image under ~2.9MB to stay safe after encoding.
_MAX_BYTES = 2_900_000

_VISION_PROMPT = (
    "Describe this figure for a technical document search index. "
    "In 2–4 short sentences, state: (1) type of visual (flowchart, architecture diagram, "
    "screenshot, chart, table image, UI mockup, etc.), (2) main components and labels you can read, "
    "(3) relationships or process flow, (4) what topic or procedure it illustrates. "
    "Use specific names from the image, not generic placeholders. "
    "If blank or unreadable, say so in one sentence."
)


def _mime_for_path(path: Path) -> str:
    low = path.suffix.lower()
    if low == ".png":
        return "image/png"
    if low in (".jpg", ".jpeg"):
        return "image/jpeg"
    if low == ".webp":
        return "image/webp"
    if low == ".gif":
        return "image/gif"
    return "image/png"


def _describe_with_vision_api(
    path: Path,
    *,
    model: str,
    api_key: str,
    base_url: str,
) -> Optional[str]:
    """OpenAI-compatible chat completions with image (Groq, OpenAI)."""
    from openai import OpenAI

    data = path.read_bytes()
    mime = _mime_for_path(path)
    url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))
    r = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
        max_tokens=320,
        temperature=0.2,
    )
    return (r.choices[0].message.content or "").strip() or None


def _groq_vision_api_key() -> str:
    dedicated = (getattr(settings, "GROQ_VISION_API_KEY", None) or "").strip()
    if dedicated:
        return dedicated
    return (getattr(settings, "GROQ_API_KEY", None) or "").strip()


def _describe_with_groq(path: Path) -> Optional[str]:
    key = _groq_vision_api_key()
    base = (getattr(settings, "GROQ_API_BASE", None) or "https://api.groq.com/openai/v1").strip()
    if not key:
        logger.warning(
            "VISION_PROVIDER=groq but GROQ_VISION_API_KEY / GROQ_API_KEY is empty"
        )
        return None
    model = (
        getattr(settings, "GROQ_VISION_MODEL", None)
        or "meta-llama/llama-4-scout-17b-16e-instruct"
    ).strip()
    return _describe_with_vision_api(path, model=model, api_key=key, base_url=base)


def _describe_with_openai(path: Path) -> Optional[str]:
    key = (getattr(settings, "OPENAI_API_KEY", None) or "").strip()
    if not key:
        logger.warning("VISION_PROVIDER=openai but OPENAI_API_KEY is empty")
        return None
    model = (getattr(settings, "VISION_CAPTION_MODEL", None) or "gpt-4o-mini").strip()
    return _describe_with_vision_api(
        path,
        model=model,
        api_key=key,
        base_url="https://api.openai.com/v1",
    )


def _describe_with_gemini(path: Path) -> Optional[str]:
    import google.generativeai as genai

    key = (getattr(settings, "GEMINI_API_KEY", None) or "").strip()
    if not key:
        logger.warning("VISION_PROVIDER=gemini but GEMINI_API_KEY is empty")
        return None
    model = (getattr(settings, "GEMINI_VISION_MODEL", None) or "gemini-2.0-flash").strip()
    genai.configure(api_key=key)
    data = path.read_bytes()
    gemini_model = genai.GenerativeModel(model)
    response = gemini_model.generate_content(
        [_VISION_PROMPT, {"mime_type": _mime_for_path(path), "data": data}],
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            max_output_tokens=320,
        ),
    )
    text = getattr(response, "text", None) or ""
    return text.strip() or None


def _describe_image(path: Path) -> Optional[str]:
    provider = (getattr(settings, "VISION_PROVIDER", None) or "groq").lower().strip()
    if provider == "openai":
        return _describe_with_openai(path)
    if provider == "gemini":
        return _describe_with_gemini(path)
    return _describe_with_groq(path)


def _vision_api_key_ok(provider: str) -> bool:
    if provider == "openai":
        return bool((getattr(settings, "OPENAI_API_KEY", None) or "").strip())
    if provider == "gemini":
        return bool((getattr(settings, "GEMINI_API_KEY", None) or "").strip())
    return bool(_groq_vision_api_key())


def enrich_image_blocks_for_search(blocks: List[Dict[str, Any]]) -> None:
    """
    Mutates image blocks in place: appends vision description to ``content`` for embedding and FTS.

    Default: Groq ``meta-llama/llama-4-scout-17b-16e-instruct`` via GROQ_API_KEY / GROQ_API_BASE.
    Cropped image files must exist under block['image_meta']['name'] (after persist_extracted_images).
    """
    if not getattr(settings, "ENABLE_VISION_IMAGE_CAPTIONS", False):
        return

    provider = (getattr(settings, "VISION_PROVIDER", None) or "groq").lower().strip()
    if not _vision_api_key_ok(provider):
        logger.warning(
            "ENABLE_VISION_IMAGE_CAPTIONS is on but API key missing for VISION_PROVIDER=%s",
            provider,
        )
        return

    max_n = max(0, int(getattr(settings, "MAX_VISION_CAPTIONS_PER_DOCUMENT", 30) or 0))
    if max_n == 0:
        return

    used = 0
    logger.info(
        "Vision captions: provider=%s max_per_doc=%s",
        provider,
        max_n,
    )

    for block in blocks:
        if block.get("block_type") != "image":
            continue
        if used >= max_n:
            logger.info(
                "Vision captions: hit MAX_VISION_CAPTIONS_PER_DOCUMENT=%s for this PDF",
                max_n,
            )
            break
        path_str = (block.get("image_meta") or {}).get("name")
        if not path_str:
            continue
        path = Path(path_str)
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > _MAX_BYTES:
                logger.warning(
                    "Vision caption skipped (file > %s bytes): %s",
                    _MAX_BYTES,
                    path,
                )
                continue
            desc = _describe_image(path)
            if desc:
                base = (block.get("content") or "").strip()
                block["content"] = (base + "\n\n[Visual description]: " + desc).strip()
                block.setdefault("image_meta", {})["vision_caption"] = desc
                block["image_meta"]["vision_provider"] = provider
                used += 1
        except Exception as e:
            logger.warning("Vision caption failed for %s: %s", path, e)

    if used:
        logger.info("Vision captions: described %s images via %s", used, provider)
