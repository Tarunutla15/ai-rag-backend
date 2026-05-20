"""DOCX text and structure extraction (paragraphs, tables, headings, code)."""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Dict, List

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.services.pdf_processor import PDFProcessor

# Approximate “page” for tree/metadata when DOCX has no real pages
_DOCX_CHARS_PER_PAGE = 2800


def _iter_block_items(parent) -> List[object]:
    """Yield paragraphs and tables in document order."""
    parent_elm = parent.element.body
    for child in parent_elm.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def _paragraph_style_is_heading(para: Paragraph) -> bool:
    name = (para.style.name if para.style else "") or ""
    low = name.lower()
    if low.startswith("heading") or low in ("title", "subtitle"):
        return True
    return False


def _paragraph_is_code(para: Paragraph) -> bool:
    text = para.text or ""
    runs = [r for r in para.runs if (r.text or "").strip()]
    if runs:
        mono = sum(
            1 for r in runs if PDFProcessor._is_monospace_font(r.font.name or "")
        )
        if len(runs) > 0 and (mono / len(runs)) >= 0.5:
            return True
    return PDFProcessor._is_code(text)


class DOCXProcessor:
    """Extract text and structure-aware blocks from Word .docx files."""

    @staticmethod
    def extract_text(docx_path: str) -> str:
        path = Path(docx_path)
        if not path.exists():
            raise FileNotFoundError(f"DOCX file not found: {docx_path}")
        doc = Document(str(path))
        parts: List[str] = []
        for block in _iter_block_items(doc):
            if isinstance(block, Paragraph):
                t = (block.text or "").strip()
                if t:
                    parts.append(t)
            elif isinstance(block, Table):
                for row in block.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        parts.append(" | ".join(cells))
        text = "\n".join(parts).strip()
        if not text:
            raise ValueError("No text could be extracted from the DOCX")
        return text

    @classmethod
    def extract_blocks(cls, docx_path: str) -> List[Dict]:
        path = Path(docx_path)
        if not path.exists():
            raise FileNotFoundError(f"DOCX file not found: {docx_path}")

        doc = Document(str(path))
        blocks: List[Dict] = []
        current_section_title = ""
        virtual_page = 1
        chars_on_page = 0

        def _bump_chars(n: int) -> None:
            nonlocal virtual_page, chars_on_page
            chars_on_page += n
            while chars_on_page >= _DOCX_CHARS_PER_PAGE:
                chars_on_page -= _DOCX_CHARS_PER_PAGE
                virtual_page += 1

        current_type: str | None = None
        buffer: List[str] = []
        pending_heading: str | None = None

        def _flush() -> None:
            nonlocal current_type, buffer, pending_heading
            if not buffer:
                return
            content = "\n".join(buffer).strip()
            if not content:
                buffer = []
                return
            block_id = str(uuid.uuid4())
            if current_type == "heading":
                nonlocal current_section_title
                current_section_title = content
                pending_heading = content
            else:
                if pending_heading and current_type == "text":
                    content = f"{pending_heading}\n{content}"
                    pending_heading = None
                blocks.append(
                    {
                        "block_id": block_id,
                        "block_type": current_type or "text",
                        "content": content,
                        "page_number": virtual_page,
                        "section_title": current_section_title,
                    }
                )
            buffer = []

        for item in _iter_block_items(doc):
            if isinstance(item, Table):
                _flush()
                current_type = None
                table_rows: List[List[str]] = []
                for row in item.rows:
                    table_rows.append([c.text.strip() for c in row.cells])
                if table_rows:
                    blocks.append(
                        {
                            "block_id": str(uuid.uuid4()),
                            "block_type": "table",
                            "content": "",
                            "table_rows": table_rows,
                            "table_summary": PDFProcessor._summarize_table(table_rows),
                            "page_number": virtual_page,
                            "section_title": current_section_title,
                        }
                    )
                    _bump_chars(200)
                continue

            para: Paragraph = item
            line = (para.text or "").strip()
            if not line:
                continue

            if _paragraph_style_is_heading(para) or PDFProcessor._is_heading(line):
                line_type = "heading"
            elif _paragraph_is_code(para):
                line_type = "code"
            else:
                line_type = "text"

            if current_type is None:
                current_type = line_type
                buffer = [line]
                _bump_chars(len(line))
                continue

            if line_type == current_type:
                buffer.append(line)
            else:
                _flush()
                current_type = line_type
                buffer = [line]
            _bump_chars(len(line))

        _flush()

        # Merge adjacent code blocks on the same virtual page
        merged: List[Dict] = []
        i = 0
        while i < len(blocks):
            b = blocks[i]
            if b.get("block_type") != "code":
                merged.append(b)
                i += 1
                continue
            code_parts = [b.get("content", "").strip()]
            j = i + 1
            while (
                j < len(blocks)
                and blocks[j].get("block_type") == "code"
                and blocks[j].get("page_number") == b.get("page_number")
            ):
                code_parts.append(blocks[j].get("content", "").strip())
                j += 1
            merged.append(
                {
                    "block_id": b.get("block_id", str(uuid.uuid4())),
                    "block_type": "code",
                    "content": "\n\n".join(p for p in code_parts if p),
                    "page_number": b.get("page_number"),
                    "section_title": b.get("section_title", ""),
                }
            )
            i = j

        for block in merged:
            if block.get("block_type") == "code":
                block["code_summary"] = PDFProcessor._summarize_code(
                    block.get("content", "")
                )

        return merged

    @staticmethod
    def persist_extracted_images(
        docx_path: str, document_id: str, blocks: List[Dict], upload_dir: str
    ) -> None:
        """DOCX image crop/export not implemented yet; ingest continues without figures."""
        return
