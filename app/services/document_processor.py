"""Unified document ingest: PDF and DOCX."""
from pathlib import Path
from typing import List, Dict, Optional

from app.config import settings
from app.services.pdf_processor import PDFProcessor
from app.services.docx_processor import DOCXProcessor

ALLOWED_EXTENSIONS = frozenset({".pdf", ".docx"})
ALLOWED_CONTENT_TYPES = frozenset({
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
})


def extension_for_filename(filename: str) -> str:
    return (Path(filename or "").suffix or "").lower()


def is_allowed_upload(filename: str) -> bool:
    return extension_for_filename(filename) in ALLOWED_EXTENSIONS


def file_type_label(ext: str) -> str:
    e = (ext or "").lower()
    if e == ".docx":
        return "docx"
    return "pdf"


def resolve_stored_document_path(
    document_id: str, upload_dir: str, registry_path: Optional[str] = None
) -> Optional[Path]:
    """Find on-disk source file for a document (pdf or docx)."""
    root = Path(upload_dir)
    for ext in ALLOWED_EXTENSIONS:
        candidate = root / f"{document_id}{ext}"
        if candidate.is_file():
            return candidate
    if registry_path:
        p = Path(registry_path)
        if p.is_file():
            return p
    return None


def media_type_for_path(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return "application/pdf"


class DocumentProcessor:
    """Route extraction to PDF or DOCX backend."""

    @staticmethod
    def extract_text(file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        if ext == ".docx":
            return DOCXProcessor.extract_text(file_path)
        if ext == ".pdf":
            return PDFProcessor.extract_text(file_path)
        raise ValueError(f"Unsupported document type: {ext}")

    @staticmethod
    def extract_blocks(file_path: str) -> List[Dict]:
        ext = Path(file_path).suffix.lower()
        if ext == ".docx":
            return DOCXProcessor.extract_blocks(file_path)
        if ext == ".pdf":
            return PDFProcessor.extract_blocks(file_path)
        raise ValueError(f"Unsupported document type: {ext}")

    @staticmethod
    def persist_extracted_images(
        file_path: str,
        document_id: str,
        blocks: List[Dict],
        upload_dir: str,
    ) -> None:
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf" and getattr(settings, "STORE_EXTRACTED_IMAGES", True):
            PDFProcessor.persist_extracted_images(
                file_path, document_id, blocks, upload_dir
            )
        elif ext == ".docx":
            DOCXProcessor.persist_extracted_images(
                file_path, document_id, blocks, upload_dir
            )
