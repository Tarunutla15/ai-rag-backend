"""Document library routes: list, view source file, replace/update, and full delete."""
from pathlib import Path
from typing import List, Optional, Dict, Any
import os
import shutil
import traceback
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

from app.config import settings
from app.models.schemas import DocumentInfo, UploadResponse
from app.services.document_store import get_document_store
from app.services.document_classifier import get_document_classifier
from app.services.document_processor import (
    DocumentProcessor,
    is_allowed_upload,
    extension_for_filename,
    resolve_stored_document_path,
    media_type_for_path,
)
from app.services.chunking import ChunkingService
from app.services.embedding import EmbeddingService
from app.services.document_tree import DocumentTreeBuilder
from app.services.document_graph_store import get_document_graph_store
from app.services.image_caption_enrichment import enrich_image_blocks_for_search
from app.services.chunk_store import get_chunk_store
from app.services.keyword_search import get_keyword_search_service
from app.services.raw_block_store import get_raw_block_store
from app.api.routes.upload import (
    get_vector_store,
    discard_failed_upload,
    best_effort_clear_prior_ingest_async,
    format_ingest_error,
    _texts_for_embedding,
)
from app.services.document_purge import (
    purge_document_everywhere_async,
    purge_many_documents_async,
    purge_index_data_async,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


def _to_document_info(doc: dict) -> DocumentInfo:
    return DocumentInfo(
        document_id=doc.get("document_id") or doc.get("document_id") or "",
        file_name=doc.get("file_name") or "",
        status=doc.get("status"),
        chunk_count=int(doc.get("chunk_count") or 0),
        technology=doc.get("technology"),
        domain=doc.get("domain"),
        created_at=doc.get("created_at"),
        updated_at=doc.get("updated_at"),
        pdf_path=doc.get("pdf_path"),
    )


@router.get("/", response_model=List[DocumentInfo])
async def list_documents(status: Optional[str] = None) -> List[DocumentInfo]:
    """List all documents in the registry."""
    store = get_document_store()
    docs = store.list_documents()
    docs = [d for d in docs if (d.get("status") or "").upper() != "FAILED"]
    if status:
        docs = [d for d in docs if (d.get("status") or "").upper() == status.upper()]
    return [_to_document_info(d) for d in docs]


@router.delete("/")
async def delete_all_documents() -> Dict[str, Any]:
    """Permanently delete every document in the library (async concurrent purge)."""
    store = get_document_store()
    docs = store.list_documents()
    doc_ids = [str(d["document_id"]) for d in docs if d.get("document_id")]
    if not doc_ids:
        return {"message": "All documents deleted", "deleted_count": 0, "errors": []}

    vector_store = get_vector_store()
    deleted_count, errors = await purge_many_documents_async(doc_ids, vector_store)
    return {
        "message": "All documents deleted",
        "deleted_count": deleted_count,
        "errors": errors,
    }


@router.get("/{document_id}", response_model=DocumentInfo)
async def get_document(document_id: str) -> DocumentInfo:
    """Get a single document's metadata."""
    store = get_document_store()
    doc = store.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return _to_document_info(doc)


@router.delete("/{document_id}")
async def delete_document(document_id: str) -> Dict[str, Any]:
    """
    Permanently delete a document: Supabase/SQLite chunks + FTS + raw blocks,
    Zilliz vectors, session scope links, local PDF + image folder, and the documents row.
    """
    vector_store = get_vector_store()
    result = await purge_document_everywhere_async(document_id, vector_store)
    if not result.get("deleted"):
        raise HTTPException(status_code=404, detail="Document not found")
    return result


@router.get("/{document_id}/pdf")
async def get_document_pdf(document_id: str):
    """Return the stored source file (PDF or DOCX) for a document_id."""
    store = get_document_store()
    doc = store.get_document(document_id) or {}
    file_path = resolve_stored_document_path(
        document_id,
        settings.UPLOAD_DIR,
        preferred_name=doc.get("file_name"),
    )
    if not file_path or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Source file not found on disk")
    return FileResponse(
        path=str(file_path),
        media_type=media_type_for_path(file_path),
        filename=doc.get("file_name") or file_path.name,
    )


@router.get("/{document_id}/images/{image_id}")
async def get_document_extracted_image(document_id: str, image_id: str):
    """
    Serve a cropped figure saved during ingest (uploads/images/{document_id}/...).
    Used by chat answers as ![caption](/documents/{document_id}/images/{image_id}).
    """
    row = get_raw_block_store().get_image(image_id)
    if not row or row.get("document_id") != document_id:
        raise HTTPException(status_code=404, detail="Image not found for this document")
    path_str = row.get("image_path") or ""
    if not path_str.strip():
        raise HTTPException(status_code=404, detail="Image has no file path")
    p = Path(path_str).resolve()
    upload_root = Path(settings.UPLOAD_DIR).resolve()
    try:
        p.relative_to(upload_root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid image path")
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Image file not on disk")
    ext = p.suffix.lower()
    media = (
        "image/png"
        if ext == ".png"
        else "image/jpeg"
        if ext in (".jpg", ".jpeg")
        else "application/octet-stream"
    )
    return FileResponse(path=str(p), media_type=media, filename=p.name)


@router.post("/{document_id}/replace", response_model=UploadResponse)
async def replace_document(document_id: str, file: UploadFile = File(...)):
    """Replace file under same document_id and re-ingest (async index purge)."""
    if not is_allowed_upload(file.filename):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are allowed")

    store = get_document_store()
    existing = store.get_document(document_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Document not found")

    file_bytes = await file.read()
    new_hash = store.compute_hash_from_bytes(file_bytes)

    dup = store.get_by_hash(new_hash)
    if dup and dup.get("document_id") != document_id:
        raise HTTPException(
            status_code=409,
            detail=f"Duplicate detected: this file already exists as document {dup.get('document_id')}",
        )

    vector_store = get_vector_store()
    await purge_index_data_async(document_id, vector_store)

    store.update_document(
        document_id,
        {
            "file_name": file.filename,
            "file_hash": new_hash,
            "status": "UPLOADED",
            "chunk_count": 0,
            "error": None,
        },
    )

    from app.utils.helpers import ensure_upload_dir, get_file_path

    ensure_upload_dir(settings.UPLOAD_DIR)
    ext = extension_for_filename(file.filename)
    stored_path = get_file_path(settings.UPLOAD_DIR, f"{document_id}{ext}")
    upload_root = Path(settings.UPLOAD_DIR)
    for other in (".pdf", ".docx"):
        if other != ext:
            old = upload_root / f"{document_id}{other}"
            if old.is_file():
                try:
                    old.unlink()
                except Exception:
                    pass
    with open(stored_path, "wb") as f:
        f.write(file_bytes)

    try:
        _chunk_size = settings.CHUNK_SIZE
        _chunk_overlap = settings.CHUNK_OVERLAP
        if getattr(settings, "CHUNK_TARGET_TOKENS", 0) > 0:
            _chunk_size = settings.CHUNK_TARGET_TOKENS * 4
            _chunk_overlap = int(_chunk_size * getattr(settings, "CHUNK_OVERLAP_PERCENT", 12) / 100)
        chunking_service = ChunkingService(chunk_size=_chunk_size, chunk_overlap=_chunk_overlap)
        embedding_service = EmbeddingService(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_EMBEDDING_MODEL,
        )

        blocks = DocumentProcessor.extract_blocks(str(stored_path))
        if getattr(settings, "STORE_EXTRACTED_IMAGES", True):
            try:
                DocumentProcessor.persist_extracted_images(
                    str(stored_path), document_id, blocks, settings.UPLOAD_DIR
                )
            except Exception as img_err:
                logger.warning("Image extraction/persist failed (replace): %s", img_err)
        try:
            enrich_image_blocks_for_search(blocks)
        except Exception as cap_err:
            logger.warning("Image vision caption enrichment failed (non-fatal): %s", cap_err)

        tree_builder = DocumentTreeBuilder()
        blocks, tree_nodes, tree_edges = tree_builder.build(blocks, document_id)

        sample_text = ""
        for b in blocks[:200]:
            sample_text += (b.get("content") or "") + "\n"
            if len(sample_text) >= 50000:
                break
        technology, domain = get_document_classifier().classify(file.filename, sample_text)

        chunks, chunk_metadata = chunking_service.chunk_blocks(
            blocks,
            document_id=document_id,
            file_name=file.filename,
            file_id=document_id,
        )
        if not chunks:
            discard_failed_upload(document_id, stored_path)
            raise HTTPException(status_code=400, detail="No text chunks could be created from document")

        await best_effort_clear_prior_ingest_async(document_id)
        chunk_store = get_chunk_store()
        chunk_ids, phase2_edges = chunk_store.prepare_chunk_ids_and_phase2_links(
            chunk_metadata, document_id, blocks=blocks
        )
        tree_edges.extend(phase2_edges)

        get_document_graph_store().insert_graph(document_id, tree_nodes, tree_edges)
        chunk_store.insert_chunks(chunks, chunk_metadata, document_id)

        embeddings = _texts_for_embedding(
            chunks,
            chunk_metadata,
            document_title=file.filename,
            embedding_service=embedding_service,
        )

        chunks_created = vector_store.insert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            document_id=document_id,
            file_hash=new_hash,
            file_name=file.filename,
            file_size=len(file_bytes),
            chunk_metadata=chunk_metadata,
            file_id=document_id,
            chunk_ids=chunk_ids,
            technology=technology,
            domain=domain,
        )

        try:
            keyword_service = get_keyword_search_service()
            keyword_service.delete_document(document_id)
            keyword_service.index_chunks(
                chunks=chunks,
                chunk_ids=chunk_ids,
                document_id=document_id,
                file_name=file.filename,
                technology=technology,
                domain=domain,
                file_id=document_id,
                start_chunk_index=0,
            )
        except Exception as ke:
            logger.warning("Keyword indexing failed during replace: %s", ke)

        store.mark_ingested(
            document_id=document_id,
            chunk_count=chunks_created,
            technology=technology,
            domain=domain,
            pdf_path=str(stored_path) if getattr(settings, "STORE_PDF_AFTER_INGEST", True) else None,
        )

        return UploadResponse(
            message="Document replaced and re-ingested successfully",
            file_id=document_id,
            chunks_created=chunks_created,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Replace ingestion failed: %s\n%s", e, traceback.format_exc())
        try:
            store.mark_failed(document_id=document_id, error=str(e))
        except Exception:
            pass
        discard_failed_upload(document_id, stored_path)
        raise HTTPException(status_code=500, detail=format_ingest_error(e))
