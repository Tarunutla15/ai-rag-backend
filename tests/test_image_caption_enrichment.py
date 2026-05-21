"""Unit tests for vision caption enrichment (Groq default)."""
from pathlib import Path
from unittest.mock import patch

from app.services import image_caption_enrichment as ice


def test_enrich_skips_when_disabled():
    blocks = [{"block_type": "image", "content": "x", "image_meta": {"name": "n.png"}}]
    with patch.object(ice.settings, "ENABLE_VISION_IMAGE_CAPTIONS", False):
        ice.enrich_image_blocks_for_search(blocks)
    assert "[Visual description]" not in blocks[0]["content"]


def test_enrich_groq_appends_caption(tmp_path):
    img = tmp_path / "fig.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    blocks = [
        {
            "block_type": "image",
            "content": "Figure 1 on page 2",
            "image_meta": {"name": str(img)},
        }
    ]
    with (
        patch.object(ice.settings, "ENABLE_VISION_IMAGE_CAPTIONS", True),
        patch.object(ice.settings, "VISION_PROVIDER", "groq"),
        patch.object(ice.settings, "GROQ_API_KEY", "gsk-test"),
        patch.object(
            ice.settings,
            "GROQ_VISION_MODEL",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ),
        patch.object(ice.settings, "MAX_VISION_CAPTIONS_PER_DOCUMENT", 10),
        patch.object(
            ice,
            "_describe_with_groq",
            return_value="Architecture diagram with API gateway.",
        ),
    ):
        ice.enrich_image_blocks_for_search(blocks)

    assert "[Visual description]: Architecture diagram" in blocks[0]["content"]
    assert blocks[0]["image_meta"]["vision_caption"] == "Architecture diagram with API gateway."
    assert blocks[0]["image_meta"]["vision_provider"] == "groq"


def test_describe_image_openai_provider(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\xff\xd8\xff")
    with (
        patch.object(ice.settings, "VISION_PROVIDER", "openai"),
        patch.object(ice.settings, "OPENAI_API_KEY", "sk-test"),
        patch.object(ice.settings, "VISION_CAPTION_MODEL", "gpt-4o-mini"),
        patch.object(ice, "_describe_with_openai", return_value="Chart of latency.") as mock_oai,
    ):
        out = ice._describe_image(img)
    assert out == "Chart of latency."
    mock_oai.assert_called_once()
