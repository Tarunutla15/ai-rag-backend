"""PDF text extraction service."""
import PyPDF2
import pdfplumber
from pathlib import Path
from typing import Optional, List, Dict
import re
import uuid


class PDFProcessor:
    """Service for extracting text from PDF files."""
    
    @staticmethod
    def extract_text(pdf_path: str, use_pdfplumber: bool = True) -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            use_pdfplumber: Use pdfplumber (better for complex PDFs) or PyPDF2
            
        Returns:
            Extracted text as a string
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        text = ""
        
        if use_pdfplumber:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                # Fallback to PyPDF2 if pdfplumber fails
                print(f"pdfplumber failed, trying PyPDF2: {e}")
                use_pdfplumber = False
        
        if not use_pdfplumber:
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e:
                raise ValueError(f"Failed to extract text from PDF: {e}")
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        return text.strip()

    @staticmethod
    def _is_heading(line: str) -> bool:
        if not line:
            return False
        stripped = line.strip()
        if len(stripped) > 80:
            return False
        if stripped.isupper() and len(stripped) >= 4:
            return True
        if re.match(r"^\d+(\.\d+)*\s+\S+", stripped):
            return True
        if stripped.endswith(":") and len(stripped.split()) <= 8:
            return True
        return False

    @staticmethod
    def _is_monospace_font(fontname: str) -> bool:
        """True if font name suggests monospace/code (e.g. Courier, Consolas, Mono)."""
        if not fontname:
            return False
        name = (fontname or "").lower()
        return any(x in name for x in ("mono", "courier", "consolas", "fixed", "source code", "menlo", "liberation mono"))

    @staticmethod
    def _is_code(line: str) -> bool:
        if not line:
            return False
        stripped = line.rstrip("\n")
        if stripped.startswith("    ") or stripped.startswith("\t"):
            return True
        code_tokens = ["{", "}", "();", "==", "!=", "<=", ">=", "::", "->"]
        if any(t in stripped for t in code_tokens):
            return True
        if re.match(r"^\s*(class|def|public|private|protected|static|void|int|float|double|String)\b", stripped):
            return True
        symbol_count = sum(1 for c in stripped if c in "{}[]();=<>")
        if len(stripped) > 0 and (symbol_count / len(stripped)) > 0.15:
            return True
        return False

    @staticmethod
    def _clearly_ends_code_block(line: str) -> bool:
        """True if this line should end a code block (blank line or heading)."""
        if not line or not line.strip():
            return True
        return PDFProcessor._is_heading(line)

    @staticmethod
    def _summarize_table(table_rows: List[List[str]]) -> str:
        if not table_rows:
            return ""
        header = table_rows[0]
        cols = [c for c in header if c]
        summary = "Table with columns: " + ", ".join(cols[:8])
        # Add a few sample rows
        samples = []
        for row in table_rows[1:4]:
            row_vals = [c for c in row if c]
            if row_vals:
                samples.append(" | ".join(row_vals[:8]))
        if samples:
            summary += ". Sample rows: " + " ; ".join(samples)
        return summary

    @staticmethod
    def _summarize_code(code_text: str) -> str:
        if not code_text:
            return ""
        # Extract function/class names
        names = re.findall(r"\b(class|def)\s+([A-Za-z_][A-Za-z0-9_]*)", code_text)
        name_list = [n[1] for n in names]
        if name_list:
            return "Code block defining: " + ", ".join(name_list[:5])
        # Fallback summary
        return "Code block with program logic and statements."

    @classmethod
    def _get_lines_with_font(cls, page) -> List[tuple]:
        """
        Build lines from page words with font info. Returns list of (line_text, is_code_from_font).
        Falls back to extract_text().splitlines() with no font info if words fail.
        """
        try:
            words = page.extract_words(extra_attrs=["fontname"])
        except Exception:
            words = []
        if not words:
            text = page.extract_text() or ""
            return [(line, False) for line in text.splitlines()]
        # Sort by vertical position then horizontal so words form lines left-to-right, top-to-bottom
        ordered = sorted(words, key=lambda w: (round((w.get("top") or 0) / 3), w.get("x0", 0)))
        lines_list: List[tuple] = []
        current_top = None
        current_words: List[Dict] = []
        for w in ordered:
            top_key = round((w.get("top") or 0) / 3)
            if current_top is not None and top_key != current_top:
                if current_words:
                    line_text = " ".join(x.get("text", "") for x in current_words)
                    mono_count = sum(1 for x in current_words if cls._is_monospace_font(x.get("fontname") or ""))
                    is_code = len(current_words) > 0 and (mono_count / len(current_words)) >= 0.5
                    lines_list.append((line_text, is_code))
                current_words = []
            current_top = top_key
            current_words.append(w)
        if current_words:
            line_text = " ".join(x.get("text", "") for x in current_words)
            mono_count = sum(1 for x in current_words if cls._is_monospace_font(x.get("fontname") or ""))
            is_code = (mono_count / len(current_words)) >= 0.5
            lines_list.append((line_text, is_code))
        if not lines_list:
            text = page.extract_text() or ""
            return [(line, False) for line in text.splitlines()]
        return lines_list

    @classmethod
    def _page_context_before_image(cls, blocks: List[Dict], page_number: int, max_chars: int = 380) -> str:
        """
        Text or table summary already placed on the same page — often captions or paragraphs
        that reference the figure (better retrieval than 'Image on page N' alone).
        """
        for b in reversed(blocks):
            if b.get("page_number") != page_number:
                continue
            if b.get("block_type") == "text":
                c = (b.get("content") or "").strip()
                if c:
                    return (c[-max_chars:] if len(c) > max_chars else c).strip()
            if b.get("block_type") == "table":
                s = (b.get("table_summary") or "").strip()
                if s:
                    return s[:max_chars].strip()
        return ""

    @classmethod
    def extract_blocks(cls, pdf_path: str) -> List[Dict]:
        """
        Extract structure-aware blocks from PDF.
        Returns list of block dicts with type and metadata.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        blocks: List[Dict] = []
        current_section_title = ""
        last_text_block_id_by_page: Dict[int, str] = {}

        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                lines_with_font = cls._get_lines_with_font(page)
                # Build text/code/heading blocks; each item is (line_text, is_code_from_font)
                current_type = None
                buffer = []
                pending_heading = None
                for line, is_code_from_font in lines_with_font:
                    line_type = "text"
                    if current_type == "code":
                        # Stay in code block until blank line or heading (keeps full code blocks together)
                        if cls._clearly_ends_code_block(line):
                            line_type = "heading" if cls._is_heading(line) else "text"
                        else:
                            line_type = "code"
                    else:
                        if cls._is_heading(line):
                            line_type = "heading"
                        elif is_code_from_font or cls._is_code(line):
                            # Layout/font: monospace → CODE; else heuristic
                            line_type = "code"

                    if current_type is None:
                        current_type = line_type
                        buffer = [line]
                        continue

                    if line_type == current_type:
                        buffer.append(line)
                    else:
                        content = "\n".join(buffer).strip()
                        if content:
                            block_id = str(uuid.uuid4())
                            if current_type == "heading":
                                current_section_title = content
                                # Hold heading to merge with next text block
                                pending_heading = content
                            else:
                                # Merge pending heading into next text block if available
                                if pending_heading and current_type == "text":
                                    content = f"{pending_heading}\n{content}"
                                    pending_heading = None
                                blocks.append({
                                    "block_id": block_id,
                                    "block_type": current_type,
                                    "content": content,
                                    "page_number": page_index,
                                    "section_title": current_section_title,
                                })
                                if current_type == "text":
                                    last_text_block_id_by_page[page_index] = block_id
                        current_type = line_type
                        buffer = [line]

                # Flush buffer
                if buffer:
                    content = "\n".join(buffer).strip()
                    if content:
                        block_id = str(uuid.uuid4())
                        if current_type == "heading":
                            current_section_title = content
                            # Keep heading pending (merge with next text block)
                            pending_heading = content
                        else:
                            if pending_heading and current_type == "text":
                                content = f"{pending_heading}\n{content}"
                                pending_heading = None
                            blocks.append({
                                "block_id": block_id,
                                "block_type": current_type or "text",
                                "content": content,
                                "page_number": page_index,
                                "section_title": current_section_title,
                            })
                            if (current_type or "text") == "text":
                                last_text_block_id_by_page[page_index] = block_id

                # Extract tables
                try:
                    tables = page.extract_tables() or []
                except Exception:
                    tables = []
                for table_rows in tables:
                    block_id = str(uuid.uuid4())
                    blocks.append({
                        "block_id": block_id,
                        "block_type": "table",
                        "content": "",
                        "table_rows": table_rows,
                        "table_summary": cls._summarize_table(table_rows),
                        "page_number": page_index,
                        "section_title": current_section_title,
                    })

                # Extract images (metadata only)
                try:
                    images = page.images or []
                except Exception:
                    images = []
                for img in images:
                    block_id = str(uuid.uuid4())
                    nearby = cls._page_context_before_image(blocks, page_index)
                    section = (current_section_title or "").strip()
                    parts = [
                        f"Figure, diagram, or raster image on PDF page {page_index}.",
                        "May be a chart, plot, architecture diagram, flowchart, screenshot, or illustration.",
                    ]
                    if section:
                        parts.append(f"Section heading: {section}.")
                    if nearby:
                        parts.append(f"Text on the same page (may label or describe this figure): {nearby}")
                    image_content = " ".join(parts)
                    img_block: Dict = {
                        "block_id": block_id,
                        "block_type": "image",
                        "content": image_content,
                        "image_meta": img,
                        "page_number": page_index,
                        "section_title": current_section_title,
                    }
                    ctx_bid = last_text_block_id_by_page.get(page_index)
                    if ctx_bid:
                        img_block["page_context_block_id"] = ctx_bid
                    blocks.append(img_block)

        # Merge adjacent code blocks on the same page (e.g. split only by a blank line inside a function)
        merged: List[Dict] = []
        i = 0
        while i < len(blocks):
            b = blocks[i]
            if b.get("block_type") != "code":
                merged.append(b)
                i += 1
                continue
            # Collect consecutive code blocks on same page into one
            code_parts = [b.get("content", "").strip()]
            j = i + 1
            while j < len(blocks) and blocks[j].get("block_type") == "code" and blocks[j].get("page_number") == b.get("page_number"):
                code_parts.append(blocks[j].get("content", "").strip())
                j += 1
            merged.append({
                "block_id": b.get("block_id", str(uuid.uuid4())),
                "block_type": "code",
                "content": "\n\n".join(p for p in code_parts if p),
                "page_number": b.get("page_number"),
                "section_title": b.get("section_title", ""),
            })
            i = j

        # Add code summaries for code blocks
        for block in merged:
            if block.get("block_type") == "code":
                block["code_summary"] = cls._summarize_code(block.get("content", ""))

        return merged

    @classmethod
    def persist_extracted_images(
        cls, pdf_path: str, document_id: str, blocks: List[Dict], upload_dir: str
    ) -> None:
        """
        For each image block, crop the page to the image bbox and save to
        upload_dir/images/document_id/page_N_idx.png. Updates block['image_meta']['name'] with the path.
        """
        import os
        base = Path(upload_dir) / "images" / document_id
        base.mkdir(parents=True, exist_ok=True)
        page_images: Dict[int, int] = {}
        with pdfplumber.open(pdf_path) as pdf:
            for block in blocks:
                if block.get("block_type") != "image":
                    continue
                meta = block.get("image_meta") or {}
                page_number = block.get("page_number", 1)
                page_index = max(0, page_number - 1)
                if page_index >= len(pdf.pages):
                    continue
                page = pdf.pages[page_index]
                x0 = meta.get("x0")
                top = meta.get("top")
                x1 = meta.get("x1")
                bottom = meta.get("bottom")
                if x0 is None or top is None or x1 is None or bottom is None:
                    continue
                idx = page_images.get(page_number, 0)
                page_images[page_number] = idx + 1
                out_name = f"page_{page_number}_{idx}.png"
                out_path = base / out_name
                try:
                    cropped = page.crop((x0, top, x1, bottom))
                    cropped.to_image(resolution=150).save(str(out_path))
                except Exception:
                    continue
                block.setdefault("image_meta", {})["name"] = str(out_path)

