"""Document upload endpoint (PDF and DOCX)."""
from pathlib import Path
from typing import List, Optional, Union
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.config import settings
from app.models.schemas import UploadResponse, BatchUploadResponse, SingleFileResult
from app.services.document_processor import DocumentProcessor, is_allowed_upload
from app.services.chunking import ChunkingService
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.document_store import get_document_store
from app.utils.helpers import ensure_upload_dir, generate_file_id, get_file_path
import os
import shutil
import traceback
import logging
import sys
from app.services.document_classifier import get_document_classifier
from app.services.keyword_search import get_keyword_search_service
from app.services.chunk_store import get_chunk_store
from app.services.raw_block_store import get_raw_block_store
from app.services.image_caption_enrichment import enrich_image_blocks_for_search
from app.services.chat_service import get_chat_service
from app.services.document_tree import DocumentTreeBuilder
from app.services.document_graph_store import get_document_graph_store
from app.services.contextual_chunk import build_embedding_texts_for_chunks
from app.services.document_purge import purge_index_data_async
from app.services.ingest_pipeline import ingest_document_async
import asyncio

logger = logging.getLogger(__name__)


def _texts_for_embedding(
    chunks: List[str],
    chunk_metadata: list,
    *,
    document_title: str,
    embedding_service: EmbeddingService,
):
    """Phase 3: embed prefixed contextual strings; display chunks stay unchanged."""
    if not getattr(settings, "ENABLE_CONTEXTUAL_EMBEDDINGS", True):
        return embedding_service.generate_embeddings(chunks)
    embedding_texts = build_embedding_texts_for_chunks(
        chunks,
        chunk_metadata,
        document_title=document_title,
    )
    print(
        f">>> STEP 7: Contextual embeddings enabled ({len(embedding_texts)} prefixed texts)",
        flush=True,
    )
    return embedding_service.generate_embeddings(embedding_texts)

router = APIRouter(prefix="/upload", tags=["upload"])
print(f"INFO: Upload router created with prefix: /upload")

# Initialize services (lazy initialization for vector_store to handle errors gracefully)
# Use print statements as fallback since logging might not be configured yet
try:
    print("INFO: Initializing services in upload route...")
    print("INFO: DocumentProcessor ready (PDF + DOCX)")
    
    _chunk_size = settings.CHUNK_SIZE
    _chunk_overlap = settings.CHUNK_OVERLAP
    if getattr(settings, "CHUNK_TARGET_TOKENS", 0) > 0:
        _chunk_size = settings.CHUNK_TARGET_TOKENS * 4
        _chunk_overlap = int(_chunk_size * getattr(settings, "CHUNK_OVERLAP_PERCENT", 12) / 100)
    chunking_service = ChunkingService(chunk_size=_chunk_size, chunk_overlap=_chunk_overlap)
    print(f"INFO: ChunkingService initialized (chunk_size={_chunk_size}, overlap={_chunk_overlap})")
    
    embedding_service = EmbeddingService(
        api_key=settings.OPENAI_API_KEY,
        model=settings.OPENAI_EMBEDDING_MODEL
    )
    print(f"INFO: EmbeddingService initialized (model={settings.OPENAI_EMBEDDING_MODEL})")
    
    document_store = get_document_store()
    print("INFO: DocumentStore initialized (DB-backed)")
except Exception as e:
    print(f"ERROR: Failed to initialize services: {type(e).__name__}: {str(e)}")
    print(traceback.format_exc())
    raise

# Initialize vector store (will be created on first use if initialization fails)
_vector_store = None

def _prepare_chunk_metadata(chunks, pdf_path: str):
    """
    Prepare metadata for each chunk (page numbers, section titles, etc.).
    
    Args:
        chunks: List of text chunks
        pdf_path: Path to the PDF file
        
    Returns:
        List of metadata dictionaries for each chunk
    """
    import pdfplumber
    
    metadata_list = []
    
    try:
        # Try to extract page information using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            # For now, we'll assign chunks to pages based on approximate position
            # This is a simple heuristic - can be improved with better PDF parsing
            total_text_length = sum(len(chunk) for chunk in chunks)
            current_pos = 0
            
            for chunk in chunks:
                chunk_meta = {
                    "file_type": "pdf",
                    "language": "en",  # Default, can be enhanced with language detection
                    "has_tables": False,
                    "has_images": False,
                    "chunk_type": "paragraph",
                    "section_title": "",
                    "page_number": -1,
                    "metadata_json": "{}"
                }
                
                # Estimate page number based on text position
                # This is approximate - better would be to track actual page breaks
                if total_text_length > 0:
                    progress = current_pos / total_text_length
                    estimated_page = int(progress * total_pages) + 1
                    chunk_meta["page_number"] = min(estimated_page, total_pages)
                
                # Check if chunk might contain table-like content (simple heuristic)
                if "\t" in chunk or "|" in chunk or chunk.count("\n") > chunk.count(" ") * 0.3:
                    chunk_meta["has_tables"] = True
                
                # Check if chunk mentions images (simple heuristic)
                image_keywords = ["figure", "image", "diagram", "chart", "graph"]
                if any(keyword in chunk.lower() for keyword in image_keywords):
                    chunk_meta["has_images"] = True
                
                # Detect chunk type (simple heuristics)
                if len(chunk) < 100 and chunk.count("\n") < 3:
                    chunk_meta["chunk_type"] = "heading"
                elif chunk.count("\n") > 10:
                    chunk_meta["chunk_type"] = "list"
                
                current_pos += len(chunk)
                metadata_list.append(chunk_meta)
                
    except Exception as e:
        print(f">>> METADATA: Error extracting metadata: {e}, using defaults", flush=True)
        # If extraction fails, use default metadata
        for chunk in chunks:
            metadata_list.append({
                "file_type": "pdf",
                "language": "en",
                "has_tables": False,
                "has_images": False,
                "chunk_type": "paragraph",
                "section_title": "",
                "page_number": -1,
                "metadata_json": "{}"
            })
    
    return metadata_list


def get_vector_store():
    """Get or initialize vector store."""
    import sys
    global _vector_store
    if _vector_store is None:
        print(f">>> VECTOR STORE: Initializing...", flush=True)
        print(f">>> VECTOR STORE: URI={settings.ZILLIZ_URI[:50]}...", flush=True)
        print(f">>> VECTOR STORE: Collection={settings.ZILLIZ_COLLECTION_NAME}", flush=True)
        sys.stdout.flush()
        try:
            _vector_store = VectorStore(
                uri=settings.ZILLIZ_URI,
                token=settings.ZILLIZ_TOKEN,
                collection_name=settings.ZILLIZ_COLLECTION_NAME
            )
            print(">>> VECTOR STORE: Initialized successfully!", flush=True)
            sys.stdout.flush()
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f">>> VECTOR STORE ERROR: {type(e).__name__}: {str(e)}", flush=True)
            print(f">>> TRACEBACK:\n{error_trace}", flush=True)
            sys.stdout.flush()
            msg = str(e)
            # Common Zilliz serverless state: cluster is paused/stopped
            if "cluster status STOPPED" in msg or "status STOPPED" in msg or "code=90153" in msg:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Vector DB (Zilliz) is STOPPED/paused. Resume the Zilliz cluster "
                        "in Zilliz Cloud, then retry upload/chat."
                    ),
                )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize vector store: {str(e)}"
            )
    return _vector_store


def format_ingest_error(exc: Exception) -> str:
    """Turn low-level DB/API errors into actionable messages for the UI."""
    raw = str(exc)
    lower = raw.lower()
    if "row-level security" in lower or "42501" in raw or "violates row-level security" in lower:
        if "document_nodes" in lower or "document_edges" in lower:
            return (
                "PDF was not saved: Supabase blocked the document structure tables "
                "(document_nodes / document_edges). Open Supabase → SQL Editor, run "
                "backend/supabase_rls_policies.sql (includes policies for those tables), then upload again."
            )
        return (
            "PDF was not saved: Supabase row-level security (RLS) blocked a database write. "
            "Run backend/supabase_rls_policies.sql in the Supabase SQL Editor, then retry."
        )
    if "document_nodes" in lower and ("does not exist" in lower or "pgrst205" in lower or "schema cache" in lower):
        return (
            "PDF was not saved: missing document_nodes table. Run backend/supabase_schema.sql "
            "in Supabase SQL Editor, then supabase_rls_policies.sql."
        )
    if len(raw) > 400:
        return raw[:400] + "…"
    return raw or "Unknown error during PDF processing"


async def best_effort_clear_prior_ingest_async(document_id: str) -> None:
    """
    Async clear of prior chunks/FTS/raw/vectors before fresh ingest.
    Does not delete document_nodes/document_edges (graph rebuilt via insert_graph).
    """
    did = (document_id or "").strip()
    if not did:
        logger.warning("best_effort_clear_prior_ingest: skipped empty document_id")
        return
    await purge_index_data_async(did, get_vector_store())


def best_effort_clear_prior_ingest(document_id: str) -> None:
    """Sync wrapper (e.g. tests). Prefer await best_effort_clear_prior_ingest_async in routes."""
    import asyncio

    asyncio.run(best_effort_clear_prior_ingest_async(document_id))


async def best_effort_clear_failed_ingest_async(document_id: str) -> None:
    """After ingest failure, remove partial index rows (async)."""
    did = (document_id or "").strip()
    if not did:
        return
    await purge_index_data_async(did, get_vector_store())


def best_effort_clear_failed_ingest(document_id: str) -> None:
    """Sync wrapper for discard_failed_upload."""
    import asyncio

    asyncio.run(best_effort_clear_failed_ingest_async(document_id))


def discard_failed_upload(
    document_id: Optional[str],
    file_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Remove all traces of a failed extract/embed/upload.
    Does not leave a FAILED row in the documents table or partial index data.
    """
    did = (document_id or "").strip()
    if not did:
        return
    best_effort_clear_failed_ingest(did)
    try:
        get_document_graph_store().delete_document_graph(did)
    except Exception as e:
        logger.warning("discard_failed_upload document_graph: %s", e)
    try:
        get_chat_service().remove_document_from_all_sessions(did)
    except Exception as e:
        logger.warning("discard_failed_upload session_documents: %s", e)
    upload_root = Path(settings.UPLOAD_DIR)
    extra = [upload_root / f"{did}{ext}" for ext in (".pdf", ".docx")]
    for p in ([file_path] if file_path else []) + extra:
        if not p:
            continue
        try:
            path = Path(p)
            if path.is_file():
                path.unlink()
        except Exception as e:
            logger.warning("discard_failed_upload pdf %s: %s", p, e)
    img_dir = upload_root / "images" / did
    try:
        if img_dir.exists() and img_dir.is_dir():
            shutil.rmtree(img_dir, ignore_errors=True)
    except Exception as e:
        logger.warning("discard_failed_upload images: %s", e)
    try:
        get_document_store().delete_document_row(did)
    except Exception as e:
        logger.warning("discard_failed_upload documents row: %s", e)


@router.post("/", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF file.
    
    - Computes file hash to prevent duplicates
    - Checks document registry for existing document
    - If new: Extracts text, chunks, generates embeddings, stores in vector database
    - If duplicate: Returns existing document info without re-processing
    """
    import sys
    print(f">>> UPLOAD ROUTE: Received file: {file.filename}", flush=True)
    sys.stdout.flush()
    
    document_id = None
    file_path = None
    
    try:
        print(">>> STEP 1: Starting PDF upload processing...", flush=True)
        sys.stdout.flush()
        # Validate file type
        if not is_allowed_upload(file.filename):
            print(f">>> VALIDATION ERROR: Invalid file type: {file.filename}", flush=True)
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files are allowed")
        
        # STEP 2: Read file content and compute hash BEFORE saving
        print(">>> STEP 2: Reading file and computing hash...", flush=True)
        file_content = await file.read()
        file_hash = document_store.compute_hash_from_bytes(file_content)
        print(f">>> STEP 2 DONE: File hash: {file_hash[:16]}...", flush=True)
        
        # STEP 3: Check for duplicate
        print(">>> STEP 3: Checking for duplicate document...", flush=True)
        existing_doc = document_store.get_by_hash(file_hash)
        if existing_doc:
            existing_doc_id = existing_doc["document_id"]
            existing_status = existing_doc.get("status")
            existing_chunks = existing_doc.get("chunk_count", 0)
            print(f">>> DUPLICATE DETECTED: Document {existing_doc_id} already exists (status: {existing_status}, chunks: {existing_chunks})", flush=True)
            
            # If already ingested, return existing document info
            if (existing_status or "").upper() == "INGESTED":
                return UploadResponse(
                    message="PDF already processed (duplicate detected). Using existing document.",
                    file_id=existing_doc_id,  # Return document_id as file_id for backward compatibility
                    chunks_created=existing_chunks
                )
            # If failed, allow re-processing
            elif (existing_status or "").upper() == "FAILED":
                print(f">>> Previous ingestion failed, re-processing document {existing_doc_id}...", flush=True)
                document_id = existing_doc_id
            # If uploaded but not ingested, continue with same document_id
            else:
                document_id = existing_doc_id
                print(f">>> Document {document_id} was uploaded but not ingested, continuing...", flush=True)
        else:
            # New document - register it
            print(">>> STEP 3a: Registering new document (DB)...", flush=True)
            document_id = document_store.create_document(
                file_name=file.filename,
                file_hash=file_hash,
                status="UPLOADED",
            )
            print(f">>> STEP 3a DONE: Registered document {document_id}", flush=True)
        
        # STEP 4: Save uploaded file
        print(">>> STEP 4: Saving uploaded file...", flush=True)
        ensure_upload_dir(settings.UPLOAD_DIR)
        file_extension = os.path.splitext(file.filename)[1]
        saved_filename = f"{document_id}{file_extension}"
        file_path = get_file_path(settings.UPLOAD_DIR, saved_filename)
        
        # Reset file pointer and save
        await file.seek(0)
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        print(f">>> STEP 4 DONE: File saved to {file_path}", flush=True)
        
        file_size = len(file_content)
        try:
            result = await ingest_document_async(
                document_id=document_id,
                file_path=str(file_path),
                file_name=file.filename,
                file_hash=file_hash,
                file_size=file_size,
                chunking_service=chunking_service,
                embedding_service=embedding_service,
                get_vector_store_fn=get_vector_store,
            )
        except ValueError as ve:
            if "No text chunks" in str(ve):
                discard_failed_upload(document_id, file_path)
                raise HTTPException(status_code=400, detail=str(ve))
            raise
        chunks_created = result.chunks_created
        technology = result.technology
        domain = result.domain
        
        print(f">>> SUCCESS: PDF processing completed for {file.filename}", flush=True)
        return UploadResponse(
            message="PDF processed and stored successfully",
            file_id=document_id,  # Return document_id as file_id for backward compatibility
            chunks_created=chunks_created
        )
    
    except HTTPException as e:
        # Re-raise HTTP exceptions as-is, but log them
        print(f">>> HTTP ERROR: {e.status_code} - {e.detail}", flush=True)
        sys.stdout.flush()
        if document_id:
            discard_failed_upload(document_id, file_path)
        raise
    except Exception as e:
        print(f">>> EXCEPTION: {type(e).__name__}: {str(e)}", flush=True)
        print(traceback.format_exc(), flush=True)
        sys.stdout.flush()
        
        if document_id:
            discard_failed_upload(document_id, file_path)

        # Log the full error for debugging
        error_trace = traceback.format_exc()
        logger.error(f"Error processing PDF: {type(e).__name__}: {str(e)}")
        logger.error(f"Full traceback:\n{error_trace}")
        
        error_detail = format_ingest_error(e)
        error_type = type(e).__name__
        
        if "Failed to initialize vector store" in str(e):
            error_detail += ". Please check your Zilliz credentials in .env file."
        elif "OpenAI" in error_detail or "API key" in error_detail or "authentication" in error_detail.lower():
            error_detail += ". Please check your OpenAI API key in .env file."
        elif "Zilliz" in error_detail or "Milvus" in error_detail:
            error_detail += ". Please check your Zilliz credentials and network connectivity."
        
        logger.error(f"Returning error to client: {error_detail}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing PDF ({error_type}): {error_detail}"
        )


@router.post("/batch", response_model=BatchUploadResponse)
async def upload_batch_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload and process multiple PDF files in batch.
    
    - Processes each file independently
    - Returns individual results for each file (success/error/duplicate)
    - Continues processing even if some files fail
    - Includes technology/domain detection for each file
    """

    
    print(f">>> BATCH UPLOAD: Received {len(files)} files", flush=True)
    sys.stdout.flush()
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate all files are PDFs
    for file in files:
        if not is_allowed_upload(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.filename}. Only PDF and DOCX files are allowed",
            )
    
    concurrent = max(1, int(getattr(settings, "BATCH_UPLOAD_CONCURRENT", 2)))
    sem = asyncio.Semaphore(concurrent)
    print(f">>> BATCH UPLOAD: concurrent workers={concurrent}", flush=True)

    async def _process_one(file: UploadFile, idx: int) -> SingleFileResult:
        async with sem:
            print(
                f"\n>>> BATCH UPLOAD: Processing file {idx}/{len(files)}: {file.filename}",
                flush=True,
            )
            document_id = None
            file_path = None
            try:
                file_content = await file.read()
                file_hash = document_store.compute_hash_from_bytes(file_content)
                existing_doc = document_store.get_by_hash(file_hash)
                if existing_doc:
                    existing_doc_id = existing_doc["document_id"]
                    existing_status = existing_doc.get("status")
                    existing_chunks = existing_doc.get("chunk_count", 0)
                    existing_tech = existing_doc.get("technology", "general")
                    existing_domain = existing_doc.get("domain", "general")
                    if (existing_status or "").upper() == "INGESTED":
                        print(f">>> DUPLICATE: {file.filename} -> {existing_doc_id}", flush=True)
                        return SingleFileResult(
                            file_name=file.filename,
                            file_id=existing_doc_id,
                            chunks_created=existing_chunks,
                            technology=existing_tech,
                            domain=existing_domain,
                            status="duplicate",
                            message="File already processed (duplicate detected)",
                        )
                    document_id = existing_doc_id
                else:
                    document_id = document_store.create_document(
                        file_name=file.filename,
                        file_hash=file_hash,
                        technology="general",
                        domain="general",
                        status="UPLOADED",
                    )

                ensure_upload_dir(settings.UPLOAD_DIR)
                file_extension = os.path.splitext(file.filename)[1]
                saved_filename = f"{document_id}{file_extension}"
                file_path = get_file_path(settings.UPLOAD_DIR, saved_filename)
                await file.seek(0)
                with open(file_path, "wb") as buffer:
                    buffer.write(file_content)

                result = await ingest_document_async(
                    document_id=document_id,
                    file_path=str(file_path),
                    file_name=file.filename,
                    file_hash=file_hash,
                    file_size=len(file_content),
                    chunking_service=chunking_service,
                    embedding_service=embedding_service,
                    get_vector_store_fn=get_vector_store,
                )
                print(
                    f">>> SUCCESS: {file.filename} ({result.chunks_created} chunks, "
                    f"{result.technology}/{result.domain})",
                    flush=True,
                )
                return SingleFileResult(
                    file_name=file.filename,
                    file_id=document_id,
                    chunks_created=result.chunks_created,
                    technology=result.technology,
                    domain=result.domain,
                    status="success",
                    message="PDF processed and stored successfully",
                )
            except ValueError as ve:
                if document_id:
                    discard_failed_upload(document_id, file_path)
                return SingleFileResult(
                    file_name=file.filename,
                    file_id=None,
                    chunks_created=0,
                    technology="general",
                    domain="general",
                    status="error",
                    message=str(ve),
                )
            except Exception as e:
                error_msg = format_ingest_error(e)
                print(f">>> ERROR processing {file.filename}: {error_msg}", flush=True)
                if document_id:
                    discard_failed_upload(document_id, file_path)
                return SingleFileResult(
                    file_name=file.filename,
                    file_id=None,
                    chunks_created=0,
                    technology="general",
                    domain="general",
                    status="error",
                    message=error_msg,
                )

    gathered = await asyncio.gather(
        *[_process_one(f, i) for i, f in enumerate(files, 1)],
        return_exceptions=True,
    )
    results: List[SingleFileResult] = []
    successful = 0
    failed = 0
    for item in gathered:
        if isinstance(item, Exception):
            failed += 1
            results.append(
                SingleFileResult(
                    file_name="unknown",
                    file_id=None,
                    chunks_created=0,
                    technology="general",
                    domain="general",
                    status="error",
                    message=str(item),
                )
            )
            continue
        results.append(item)
        if item.status == "success":
            successful += 1
        elif item.status == "error":
            failed += 1
    
    # Summary
    print(f"\n>>> BATCH UPLOAD COMPLETE: {successful} successful, {failed} failed, {len(files) - successful - failed} duplicates", flush=True)
    sys.stdout.flush()

    summary = f"Batch upload completed: {successful} successful, {failed} failed"
    if failed and not successful:
        first_err = next((r.message for r in results if r.status == "error" and r.message), None)
        if first_err:
            summary = first_err
    
    return BatchUploadResponse(
        message=summary,
        total_files=len(files),
        successful=successful,
        failed=failed,
        results=results
    )
