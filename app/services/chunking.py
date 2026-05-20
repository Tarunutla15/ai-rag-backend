"""Text chunking service."""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Tuple, Dict, Optional
from app.services.raw_block_store import get_raw_block_store
from app.services.document_tree import merge_tree_fields_into_chunk_meta


class ChunkingService:
    """Service for splitting text into chunks."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize chunking service.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = self.text_splitter.split_text(text)
        return chunks

    def chunk_blocks(
        self,
        blocks: List[Dict],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        document_id: Optional[str] = None,
        file_name: Optional[str] = None,
        file_id: Optional[str] = None,
    ) -> Tuple[List[str], List[Dict]]:
        """
        Chunk structured blocks with per-block rules.
        
        Returns:
            (chunks, metadata_list)
        """
        if not blocks:
            return [], []

        # Allow custom sizing for text blocks
        splitter = self.text_splitter
        if chunk_size or chunk_overlap:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size or 1000,
                chunk_overlap=chunk_overlap or 200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )

        raw_store = get_raw_block_store()
        chunks: List[str] = []
        metadata_list: List[Dict] = []

        source_ext = (file_name or "").lower().rsplit(".", 1)[-1] if file_name and "." in file_name else "pdf"
        source_file_type = "docx" if source_ext == "docx" else "pdf"

        for block in blocks:
            block_type = block.get("block_type", "text")
            content = (block.get("content") or "").strip()
            page_number = block.get("page_number", -1)
            section_title = block.get("section_title", "")
            meta_base = {
                "file_type": source_file_type,
                "language": "en",
                "has_tables": block_type == "table",
                "has_images": block_type == "image",
                "page_number": page_number,
                "section_title": section_title,
                "document_id": document_id,
                "file_name": file_name,
                "file_id": file_id,
            }
            if block.get("block_id"):
                meta_base["source_block_id"] = block["block_id"]
            if block.get("page_context_block_id"):
                meta_base["page_context_block_id"] = block["page_context_block_id"]

            if block_type == "heading":
                if content:
                    chunks.append(content)
                    meta = dict(meta_base)
                    meta["chunk_type"] = "heading"
                    merge_tree_fields_into_chunk_meta(block, meta)
                    metadata_list.append(meta)
                continue

            if block_type == "text":
                if not content:
                    continue
                parts = splitter.split_text(content)
                for part in parts:
                    meta = dict(meta_base)
                    meta["chunk_type"] = "paragraph"
                    merge_tree_fields_into_chunk_meta(block, meta)
                    metadata_list.append(meta)
                    chunks.append(part)
                continue

            if block_type == "table":
                table_rows = block.get("table_rows") or []
                raw_table_id = None
                if table_rows:
                    raw_table_id = raw_store.store_table(table_rows, meta_base)
                summary = (block.get("table_summary") or "").strip()
                # Build searchable text: summary + multi-row preview (up to 800 chars)
                if summary:
                    searchable = summary
                else:
                    searchable = f"Table on page {page_number}."
                if table_rows:
                    preview_parts = []
                    for row in table_rows[:4]:  # first 4 rows for better retrieval
                        cell_str = " | ".join(str(c) for c in (row[:10] if row else []) if c)
                        if cell_str:
                            preview_parts.append(cell_str)
                    preview = "\n".join(preview_parts)[:800]
                    if preview:
                        searchable = searchable + "\n" + preview
                chunks.append(searchable)
                meta = dict(meta_base)
                meta["chunk_type"] = "table_summary"
                if raw_table_id:
                    meta["raw_table_id"] = raw_table_id
                merge_tree_fields_into_chunk_meta(block, meta)
                metadata_list.append(meta)
                continue

            if block_type == "code":
                if not content:
                    continue  # skip empty code blocks (no orphaned raw row, no chunk)
                raw_code_id = raw_store.store_code(content, meta_base)
                summary = (block.get("code_summary") or "").strip()
                if summary:
                    searchable = summary
                else:
                    searchable = "Code block (program code)."
                # First 1500 chars of code so keywords in longer blocks are findable
                code_preview = content[:1500].strip()
                if code_preview:
                    searchable = searchable + "\n\n" + code_preview
                chunks.append(searchable)
                meta = dict(meta_base)
                meta["chunk_type"] = "code_summary"
                meta["raw_code_id"] = raw_code_id
                merge_tree_fields_into_chunk_meta(block, meta)
                metadata_list.append(meta)
                continue

            if block_type == "image":
                caption = content or f"Image on page {page_number}"
                meta_for_image = dict(meta_base)
                meta_for_image["caption"] = caption
                raw_image_id = raw_store.store_image(block.get("image_meta", {}), meta_for_image)
                chunks.append(caption)
                meta = dict(meta_base)
                meta["chunk_type"] = "image_caption"
                meta["raw_image_id"] = raw_image_id
                meta["caption"] = caption
                merge_tree_fields_into_chunk_meta(block, meta)
                metadata_list.append(meta)
                continue

        return chunks, metadata_list

