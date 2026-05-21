"""
Async document ingest pipeline.

CPU-heavy steps (extract, chunk, tree) run in a thread pool; network I/O
(embeddings, Zilliz insert, keyword FTS) uses async clients / parallel awaits.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from app.config import settings
from app.services.chunking import ChunkingService
from app.services.contextual_chunk import build_embedding_texts_for_chunks
from app.services.document_classifier import get_document_classifier
from app.services.document_graph_store import get_document_graph_store
from app.services.document_processor import DocumentProcessor
from app.services.document_store import get_document_store
from app.services.document_tree import DocumentTreeBuilder
from app.services.embedding import EmbeddingService
from app.services.image_caption_enrichment import enrich_image_blocks_for_search
from app.services.chunk_store import get_chunk_store
from app.services.keyword_search import get_keyword_search_service
from app.services.document_purge import purge_index_data_async
from app.services.parent_child_chunks import build_section_parents, split_searchable_for_index

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    document_id: str
    chunks_created: int
    technology: str
    domain: str


async def _texts_for_embedding_async(
    chunks: List[str],
    chunk_metadata: list,
    *,
    document_title: str,
    embedding_service: EmbeddingService,
) -> List[List[float]]:
    if not getattr(settings, "ENABLE_CONTEXTUAL_EMBEDDINGS", True):
        return await embedding_service.generate_embeddings_async(chunks)
    embedding_texts = build_embedding_texts_for_chunks(
        chunks,
        chunk_metadata,
        document_title=document_title,
    )
    print(
        f">>> STEP 7: Contextual embeddings (async) — {len(embedding_texts)} prefixed texts",
        flush=True,
    )
    return await embedding_service.generate_embeddings_async(embedding_texts)


async def ingest_document_async(
    *,
    document_id: str,
    file_path: str,
    file_name: str,
    file_hash: str,
    file_size: int,
    chunking_service: ChunkingService,
    embedding_service: EmbeddingService,
    get_vector_store_fn,
) -> IngestResult:
    """
    Full ingest for one saved file on disk (extract → index).

    Caller must register the document row and save bytes to disk before calling.
    """
    print(">>> STEP 5: Extracting structured blocks (async/thread)...", flush=True)

    def _extract_and_enrich():
        blocks = DocumentProcessor.extract_blocks(str(file_path))
        if getattr(settings, "STORE_EXTRACTED_IMAGES", True):
            try:
                DocumentProcessor.persist_extracted_images(
                    str(file_path), document_id, blocks, settings.UPLOAD_DIR
                )
            except Exception as img_err:
                logger.warning("Image extraction/persist failed: %s", img_err)
        try:
            enrich_image_blocks_for_search(blocks)
        except Exception as cap_err:
            logger.warning("Image caption enrichment failed (non-fatal): %s", cap_err)
        return blocks

    blocks = await asyncio.to_thread(_extract_and_enrich)
    print(f">>> STEP 5 DONE: Extracted {len(blocks)} blocks", flush=True)

    def _build_tree():
        builder = DocumentTreeBuilder()
        return builder.build(blocks, document_id)

    print(">>> STEP 5a: Building document tree (async/thread)...", flush=True)
    blocks, tree_nodes, tree_edges = await asyncio.to_thread(_build_tree)
    print(f">>> STEP 5a DONE: {len(tree_nodes)} nodes, {len(tree_edges)} edges", flush=True)

    sample_text = ""
    for b in blocks[:200]:
        sample_text += (b.get("content") or "") + "\n"
        if len(sample_text) >= 50000:
            break

    def _classify():
        return get_document_classifier().classify(file_name, sample_text)

    technology, domain = await asyncio.to_thread(_classify)
    print(f">>> Classified {file_name} as: {technology}/{domain}", flush=True)

    def _chunk():
        return chunking_service.chunk_blocks(
            blocks,
            document_id=document_id,
            file_name=file_name,
            file_id=document_id,
        )

    print(">>> STEP 6: Chunking blocks (async/thread)...", flush=True)
    chunks, chunk_metadata = await asyncio.to_thread(_chunk)
    print(f">>> STEP 6 DONE: Created {len(chunks)} chunks", flush=True)
    if not chunks:
        raise ValueError("No text chunks could be created from document")

    print(">>> STEP 6b: Phase 2 cross-type links...", flush=True)
    await purge_index_data_async(document_id, get_vector_store_fn())

    def _persist_graph_and_chunks():
        chunk_store = get_chunk_store()
        chunk_ids, phase2_edges = chunk_store.prepare_chunk_ids_and_phase2_links(
            chunk_metadata, document_id, blocks=blocks
        )
        parent_texts, parent_metas = build_section_parents(
            chunks,
            chunk_metadata,
            document_id=document_id,
            file_name=file_name,
        )
        all_chunks = list(chunks) + parent_texts
        all_metadata = list(chunk_metadata) + parent_metas
        all_edges = list(tree_edges)
        all_edges.extend(phase2_edges)
        get_document_graph_store().insert_graph(document_id, tree_nodes, all_edges)
        chunk_store.insert_chunks(all_chunks, all_metadata, document_id)
        return chunk_ids, phase2_edges, parent_texts

    chunk_ids, phase2_edges, parent_texts = await asyncio.to_thread(_persist_graph_and_chunks)
    print(
        f">>> STEP 6b DONE: {len(chunk_ids)} child ids, {len(parent_texts)} section parents, "
        f"{len(phase2_edges)} cross-type edges",
        flush=True,
    )

    search_chunks, search_metadata, search_chunk_ids = split_searchable_for_index(
        chunks, chunk_metadata
    )

    print(">>> STEP 7: Generating embeddings (async, children only)...", flush=True)
    embeddings = await _texts_for_embedding_async(
        search_chunks,
        search_metadata,
        document_title=file_name,
        embedding_service=embedding_service,
    )
    print(f">>> STEP 7 DONE: Generated {len(embeddings)} embeddings", flush=True)

    print(">>> STEP 8: Inserting into vector store (async, children only)...", flush=True)
    vector_store = get_vector_store_fn()
    chunks_created = await vector_store.insert_chunks_async(
        chunks=search_chunks,
        embeddings=embeddings,
        document_id=document_id,
        file_hash=file_hash,
        file_name=file_name,
        file_size=file_size,
        chunk_metadata=search_metadata,
        file_id=document_id,
        chunk_ids=search_chunk_ids,
        technology=technology,
        domain=domain,
    )
    print(f">>> STEP 8 DONE: Inserted {chunks_created} chunks", flush=True)

    async def _keyword_index():
        keyword_service = get_keyword_search_service()
        await asyncio.to_thread(keyword_service.delete_document, document_id)
        await asyncio.to_thread(
            lambda: keyword_service.index_chunks(
                chunks=search_chunks,
                chunk_ids=search_chunk_ids,
                document_id=document_id,
                file_name=file_name,
                technology=technology,
                domain=domain,
                file_id=document_id,
                start_chunk_index=0,
                chunk_metadata=search_metadata,
                document_title=file_name,
            )
        )

    try:
        print(">>> STEP 8b: Keyword FTS index (async/thread)...", flush=True)
        await _keyword_index()
        print(f">>> STEP 8b DONE: Indexed {len(search_chunks)} chunks for keyword search", flush=True)
    except Exception as e:
        logger.warning("Keyword indexing failed: %s", e)

    def _mark_ingested():
        pdf_path_for_registry = None
        if getattr(settings, "STORE_PDF_AFTER_INGEST", True):
            try:
                from app.services.blob_storage import upload_document_assets

                pdf_path_for_registry = upload_document_assets(
                    document_id,
                    file_path,
                    settings.UPLOAD_DIR,
                    blocks,
                )
            except Exception as up_err:
                logger.warning("Supabase asset upload failed: %s", up_err)
            if not pdf_path_for_registry:
                pdf_path_for_registry = str(file_path)
        get_document_store().mark_ingested(
            document_id=document_id,
            chunk_count=chunks_created,
            technology=technology,
            domain=domain,
            pdf_path=pdf_path_for_registry,
        )
        if (
            not getattr(settings, "STORE_PDF_AFTER_INGEST", True)
            and file_path
            and os.path.exists(file_path)
        ):
            try:
                os.remove(file_path)
            except Exception as rm_err:
                logger.warning("Failed to remove file after ingest: %s", rm_err)

    await asyncio.to_thread(_mark_ingested)
    print(f">>> STEP 9 DONE: Document {document_id} marked as ingested", flush=True)

    return IngestResult(
        document_id=document_id,
        chunks_created=chunks_created,
        technology=technology,
        domain=domain,
    )
