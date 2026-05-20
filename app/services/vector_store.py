"""
Zilliz Vector Store using direct REST API.
All operations (collection creation, insert, search, delete) use REST API for reliability.
Supports intelligent query routing with technology and domain filters.
"""

import asyncio
import httpx
import json
import time
from typing import List, Dict, Optional
import logging

from app.config import settings

logger = logging.getLogger(__name__)


def _zilliz_insert_timeout() -> httpx.Timeout:
    sec = float(getattr(settings, "ZILLIZ_INSERT_TIMEOUT_SECONDS", 300.0))
    return httpx.Timeout(connect=30.0, read=sec, write=sec, pool=30.0)


def _zilliz_delete_timeout() -> float:
    return float(getattr(settings, "ZILLIZ_DELETE_TIMEOUT_SECONDS", 60.0))


def _default_insert_batch_size() -> int:
    return int(getattr(settings, "ZILLIZ_INSERT_BATCH_SIZE", 40))


# --- Typed collections (chunk-type routing) ---------------------------------
# Zilliz Serverless caps collections per DB (often 5 total). Image captions are embedded
# text — they share the text_chunks collection so we only need 3 typed collections
# (text_chunks, code_blocks, tables), leaving room for a legacy collection name if present.
TEXT_SUFFIX = "text_chunks"
CODE_SUFFIX = "code_blocks"
TABLES_SUFFIX = "tables"

_BUCKET_ORDER = ("text", "code", "tables")


def _build_search_filter_expr(
    *,
    technology: Optional[str] = None,
    domain: Optional[str] = None,
    document_id: Optional[str] = None,
    file_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
) -> Optional[str]:
    """Build Zilliz boolean filter for search/delete."""
    filters: List[str] = []
    if technology:
        filters.append(f'technology == "{technology}"')
    if domain:
        filters.append(f'domain == "{domain}"')
    if document_ids:
        ids = [f'"{str(d).strip()}"' for d in document_ids if d and str(d).strip()]
        if len(ids) == 1:
            filters.append(f"document_id == {ids[0]}")
        elif ids:
            filters.append(f"document_id in [{', '.join(ids)}]")
    elif document_id:
        filters.append(f'document_id == "{document_id}"')
    elif file_id:
        filters.append(f'file_id == "{file_id}"')
    return " and ".join(filters) if filters else None


def chunk_type_to_bucket(chunk_type: Optional[str]) -> str:
    """Map chunk metadata chunk_type to a routing bucket name."""
    ct = (chunk_type or "paragraph").lower().strip()
    if ct in ("code_summary", "code"):
        return "code"
    if ct in ("table_summary", "table"):
        return "tables"
    # image_caption uses text collection (caption embeddings + chunk_type in metadata)
    if ct in ("image_caption", "image"):
        return "text"
    return "text"


def infer_retrieval_bucket_weights(query: str) -> Dict[str, float]:
    """
    Heuristic weights for text vs code vs tables (sum to 1).
    Diagram/image-style questions allocate more top_k to the text collection (captions live there).
    """
    q = (query or "").lower()
    code_signals = (
        " def ", "def ", "class ", "function ", "snippet", "syntax", "implement",
        "code ", " traceback", "traceback", "stack trace", "compile error",
        "exception ", " import ", "from ", " lambda ", "decorator",
    )
    table_signals = ("table", "row", "column", "csv", "matrix", "spreadsheet")
    image_signals = ("diagram", "figure", "image", "picture", "chart", "plot", "screenshot", "illustration")

    c = sum(1 for s in code_signals if s in q)
    t = sum(1 for s in table_signals if s in q)
    img = sum(1 for s in image_signals if s in q)

    wt, wc, wtb = 0.46, 0.28, 0.26
    if c >= 1 and c >= t and c >= img:
        wt, wc, wtb = 0.22, 0.48, 0.30
    elif t >= 1 and t >= c:
        wt, wc, wtb = 0.28, 0.18, 0.54
    elif img >= 1 and img >= c:
        wt, wc, wtb = 0.58, 0.15, 0.27

    total = wt + wc + wtb
    if total <= 0:
        return {"text": 1.0, "code": 0.0, "tables": 0.0}
    return {
        "text": wt / total,
        "code": wc / total,
        "tables": wtb / total,
    }


def _normalize_similarity_score(score: float) -> float:
    try:
        s = float(score)
    except Exception:
        return 0.0
    if s < 0:
        return 0.0
    if s <= 1.0:
        return s
    return 1.0 / (1.0 + s)


def _allocate_bucket_top_k(total_k: int, weights: Dict[str, float]) -> Dict[str, int]:
    """Largest-remainder allocation of total_k across text/code/tables buckets."""
    if total_k <= 0:
        return {k: 0 for k in _BUCKET_ORDER}
    raw = {k: total_k * weights.get(k, 0.0) for k in _BUCKET_ORDER}
    floors = {k: int(raw[k]) for k in _BUCKET_ORDER}
    rem = total_k - sum(floors.values())
    frac_order = sorted(
        _BUCKET_ORDER,
        key=lambda k: raw[k] - floors[k],
        reverse=True,
    )
    i = 0
    while rem > 0:
        floors[frac_order[i % len(frac_order)]] += 1
        rem -= 1
        i += 1
    return floors


def _format_zilliz_search_hits(result_data: dict) -> List[Dict]:
    """Parse Zilliz search JSON into normalized chunk dicts."""
    if result_data.get("code") != 0:
        logger.error(f"Search failed: {result_data}")
        return []
    formatted: List[Dict] = []
    if "data" not in result_data:
        return formatted
    for hit in result_data["data"]:
        chunk_id = hit.get("chunk_id", "")
        metadata_json = hit.get("metadata_json", "")
        if not chunk_id and metadata_json:
            try:
                meta_obj = json.loads(metadata_json)
                chunk_id = meta_obj.get("chunk_id", "") if isinstance(meta_obj, dict) else ""
            except Exception:
                chunk_id = ""
        formatted.append({
            "text": hit.get("text", ""),
            "document_id": hit.get("document_id", ""),
            "file_name": hit.get("file_name", ""),
            "technology": hit.get("technology", "general"),
            "domain": hit.get("domain", "general"),
            "chunk_index": hit.get("chunk_index", 0),
            "page_number": hit.get("page_number", -1),
            "section_title": hit.get("section_title", ""),
            "chunk_type": hit.get("chunk_type", "paragraph"),
            "metadata_json": hit.get("metadata_json", ""),
            "chunk_id": chunk_id,
            "score": hit.get("distance", hit.get("score", 0.0)),
        })
    return formatted


def _build_search_payload(
    collection_name: str,
    query_embedding: List[float],
    top_k: int,
    expr: Optional[str],
) -> dict:
    payload = {
        "collectionName": collection_name,
        "data": [query_embedding],
        "annsField": "vector",
        "limit": top_k,
        "outputFields": [
            "text",
            "document_id",
            "file_name",
            "technology",
            "domain",
            "chunk_index",
            "page_number",
            "section_title",
            "chunk_type",
            "metadata_json",
            "chunk_id",
        ],
    }
    if expr:
        payload["filter"] = expr
    return payload


class _ZillizCollection:
    """
    Client for **one** Zilliz collection name (one index). ``VectorStore`` composes **three**
    of these (text_chunks, code_blocks, tables) and merges search results — chat never uses
    only this class alone for retrieval.
    """

    VECTOR_DIMENSION = 1536
    MAX_TEXT_LEN = 65535

    def __init__(self, uri: str, token: str, collection_name: str):
        # Extract base URL - Zilliz serverless format: https://xxx.serverless.xxx.cloud.zilliz.com
        self.base_uri = uri.rstrip("/")
        # Remove any existing API paths - keep just the base domain
        for path in ["/api/v1", "/v1", "/api", "/v2/vectordb", "/v2"]:
            if path in self.base_uri:
                self.base_uri = self.base_uri.replace(path, "")
        
        # Zilliz Cloud REST API uses /v2/vectordb/entities/{operation} format
        self.api_base = f"{self.base_uri}/v2/vectordb"
        
        self.token = token
        self.collection_name = collection_name
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        self._ensure_collection()

    def _get_api_url(self, operation: str) -> str:
        """Build full API URL for Zilliz REST API entities operations.
        
        Operations: insert, search, query, get, delete, upsert
        """
        return f"{self.api_base}/entities/{operation}"

    def _ensure_collection(self):
        """Ensure collection exists using REST API for both checking and creation.
        
        Creates collection with explicit schema fields including technology and domain
        for intelligent query routing.
        """
        # First, check if collection exists using REST API
        try:
            check_url = f"{self.api_base}/collections/describe"
            check_payload = {"collectionName": self.collection_name}
            response = httpx.post(
                check_url,
                headers=self.headers,
                json=check_payload,
                timeout=30.0,
            )
            result = response.json()
            
            # If code is 0, collection exists
            if result.get("code") == 0:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return
            
            logger.info(f"Collection check response: code={result.get('code')}, message={result.get('message', 'N/A')}")
        except Exception as e:
            logger.info(f"Collection check via REST API failed: {e}, will attempt to create...")
        
        # Collection doesn't exist, create it with explicit schema using REST API
        logger.info(f"Collection '{self.collection_name}' not found, creating with explicit schema...")
        
        try:
            # Define explicit schema with all fields including technology and domain
            schema = {
                "autoId": True,
                "enableDynamicField": False,
                "fields": [
                    # Primary key
                    {
                        "fieldName": "id",
                        "dataType": "Int64",
                        "isPrimary": True,
                        "autoId": True
                    },
                    # Vector embedding
                    {
                        "fieldName": "vector",
                        "dataType": "FloatVector",
                        "elementTypeParams": {
                            "dim": str(self.VECTOR_DIMENSION)
                        }
                    },
                    # Text content
                    {
                        "fieldName": "text",
                        "dataType": "VarChar",
                        "elementTypeParams": {
                            "max_length": str(self.MAX_TEXT_LEN)
                        }
                    },
                    {
                        "fieldName": "text_length",
                        "dataType": "Int64"
                    },
                    # Document identification
                    {
                        "fieldName": "document_id",
                        "dataType": "VarChar",
                        "elementTypeParams": {
                            "max_length": "255"
                        }
                    },
                    {
                        "fieldName": "file_name",
                        "dataType": "VarChar",
                        "elementTypeParams": {
                            "max_length": "500"
                        }
                    },
                    {
                        "fieldName": "file_hash",
                        "dataType": "VarChar",
                        "elementTypeParams": {
                            "max_length": "64"
                        }
                    },
                    {
                        "fieldName": "file_id",
                        "dataType": "VarChar",
                        "elementTypeParams": {
                            "max_length": "255"
                        }
                    },
                    {
                        "fieldName": "chunk_id",
                        "dataType": "VarChar",
                        "elementTypeParams": {
                            "max_length": "64"
                        }
                    },
                    # ===== NEW: Technology and Domain for intelligent routing =====
                    {
                        "fieldName": "technology",
                        "dataType": "VarChar",
                        "elementTypeParams": {
                            "max_length": "50"
                        }
                    },
                    {
                        "fieldName": "domain",
                        "dataType": "VarChar",
                        "elementTypeParams": {
                            "max_length": "50"
                        }
                    },
                    # =============================================================
                    # Chunk positioning
                    {
                        "fieldName": "chunk_index",
                        "dataType": "Int64"
                    },
                    {
                        "fieldName": "chunk_start_char",
                        "dataType": "Int64"
                    },
                    {
                        "fieldName": "chunk_end_char",
                        "dataType": "Int64"
                    },
                    {
                        "fieldName": "total_chunks_in_doc",
                        "dataType": "Int64"
                    },
                    # Timestamps
                    {
                        "fieldName": "created_at",
                        "dataType": "Int64"
                    },
                    {
                        "fieldName": "updated_at",
                        "dataType": "Int64"
                    },
                    # Content metadata
                    {
                        "fieldName": "page_number",
                        "dataType": "Int64"
                    },
                    {
                        "fieldName": "section_title",
                        "dataType": "VarChar",
                        "elementTypeParams": {
                            "max_length": "500"
                        }
                    },
                    {
                        "fieldName": "chunk_type",
                        "dataType": "VarChar",
                        "elementTypeParams": {
                            "max_length": "50"
                        }
                    },
                    {
                        "fieldName": "file_size",
                        "dataType": "Int64"
                    },
                    {
                        "fieldName": "file_type",
                        "dataType": "VarChar",
                        "elementTypeParams": {
                            "max_length": "20"
                        }
                    },
                    {
                        "fieldName": "language",
                        "dataType": "VarChar",
                        "elementTypeParams": {
                            "max_length": "10"
                        }
                    },
                    {
                        "fieldName": "has_tables",
                        "dataType": "Int64"
                    },
                    {
                        "fieldName": "has_images",
                        "dataType": "Int64"
                    },
                    {
                        "fieldName": "metadata_json",
                        "dataType": "VarChar",
                        "elementTypeParams": {
                            "max_length": "10000"
                        }
                    }
                ]
            }
            
            # Index params for vector field
            index_params = [
                {
                    "fieldName": "vector",
                    "indexName": "vector_index",
                    "metricType": "COSINE",
                    "indexType": "AUTOINDEX",
                    "params": {}
                }
            ]
            
            # Create collection request
            create_url = f"{self.api_base}/collections/create"
            create_payload = {
                "collectionName": self.collection_name,
                "schema": schema,
                "indexParams": index_params
            }
            
            response = httpx.post(
                create_url,
                headers=self.headers,
                json=create_payload,
                timeout=60.0,
            )
            result = response.json()
            
            if result.get("code") == 0:
                logger.info(f"Collection '{self.collection_name}' created successfully with explicit schema")
            else:
                # Check if it's an "already exists" error (race condition)
                error_msg = result.get("message", "")
                if "already exist" in error_msg.lower():
                    logger.info(f"Collection '{self.collection_name}' was created by another process")
                    return
                
                raise RuntimeError(f"Failed to create collection: code={result.get('code')}, message={error_msg}")
                
        except Exception as e:
            if "already exist" in str(e).lower():
                logger.info(f"Collection '{self.collection_name}' was created by another process")
                return
            
            logger.error(f"Failed to create collection: {e}")
            raise RuntimeError(
                f"Failed to create/load collection '{self.collection_name}'. "
                f"Error: {e}"
            )

    def insert_chunks(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        document_id: str,
        file_hash: str,
        file_name: str,
        file_size: int,
        technology: str = "general",
        domain: str = "general",
        chunk_metadata: Optional[List[Dict]] = None,
        file_id: Optional[str] = None,
        chunk_ids: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        chunk_indices: Optional[List[int]] = None,
        chunk_char_starts: Optional[List[int]] = None,
    ) -> int:
        """
        Insert document chunks with embeddings into the vector store.
        Processes in batches to avoid request size limits (413 errors).
        
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            document_id: Unique document identifier
            file_hash: SHA256 hash of the file
            file_name: Original filename
            file_size: Size of the file in bytes
            technology: Technology category (e.g., 'react', 'python', 'java')
            domain: Domain category (e.g., 'frontend', 'backend', 'data-science')
            chunk_metadata: Optional list of metadata dicts per chunk
            file_id: Optional file identifier
            batch_size: Chunks per request; default from settings ZILLIZ_INSERT_BATCH_SIZE (smaller reduces timeouts)
            
        Returns:
            Number of chunks inserted
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings length mismatch")
        if chunk_ids and len(chunk_ids) != len(chunks):
            raise ValueError("Chunks and chunk_ids length mismatch")

        if batch_size is None:
            batch_size = _default_insert_batch_size()

        total_chunks = len(chunks)

        if chunk_indices is not None and len(chunk_indices) != len(chunks):
            raise ValueError("chunk_indices length mismatch")
        if chunk_char_starts is not None and len(chunk_char_starts) != len(chunks):
            raise ValueError("chunk_char_starts length mismatch")
        
        # If small number of chunks, insert all at once
        if total_chunks <= batch_size:
            return self._insert_single_batch(
                chunks, embeddings, document_id, file_hash, file_name,
                file_size, technology, domain, chunk_metadata, file_id, chunk_ids,
                start_chunk_index=0,
                start_char_pos=0,
                total_chunks_in_doc=total_chunks,
                chunk_indices=chunk_indices,
                chunk_char_starts=chunk_char_starts,
            )
        
        # Process in batches for large PDFs
        total_inserted = 0
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        print(f">>> VECTOR STORE: Inserting {total_chunks} chunks in {total_batches} batches (batch_size={batch_size})", flush=True)
        
        # Track character position across all chunks for accurate indexing
        global_char_pos = 0
        
        for i in range(0, total_chunks, batch_size):
            batch_num = i // batch_size + 1
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadata = chunk_metadata[i:i + batch_size] if chunk_metadata else None
            batch_chunk_ids = chunk_ids[i:i + batch_size] if chunk_ids else None
            batch_chunk_indices = chunk_indices[i:i + batch_size] if chunk_indices else None
            batch_char_starts = chunk_char_starts[i:i + batch_size] if chunk_char_starts else None
            
            print(f">>> VECTOR STORE: Insert batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)", flush=True)
            
            try:
                inserted = self._insert_single_batch(
                    batch_chunks, batch_embeddings, document_id, file_hash, file_name,
                    file_size, technology, domain, batch_metadata, file_id, batch_chunk_ids,
                    start_chunk_index=i, start_char_pos=global_char_pos, total_chunks_in_doc=total_chunks,
                    chunk_indices=batch_chunk_indices,
                    chunk_char_starts=batch_char_starts,
                )
                total_inserted += inserted
                
                # Update global character position for next batch
                for chunk in batch_chunks:
                    global_char_pos += len(chunk)
                
                # Small delay to avoid rate limiting
                if batch_num < total_batches:
                    time.sleep(0.2)
                    
            except Exception as e:
                error_msg = str(e)
                is_timeout = (
                    "timeout" in error_msg.lower()
                    or "timed out" in error_msg.lower()
                    or type(e).__name__ in ("WriteTimeout", "ReadTimeout", "ConnectTimeout")
                )

                # Payload too large or slow write: retry with smaller batches
                if "413" in error_msg or "entity too large" in error_msg.lower() or is_timeout:
                    reason = "too large" if "413" in error_msg or "entity too large" in error_msg.lower() else "timeout/slow"
                    print(
                        f">>> VECTOR STORE: Batch {batch_num} {reason} ({len(batch_chunks)} chunks). Retrying with smaller batches...",
                        flush=True,
                    )
                    
                    # Split this batch in half and retry
                    half_size = len(batch_chunks) // 2
                    if half_size > 0:
                        # Retry first half
                        try:
                            first_half_chunks = batch_chunks[:half_size]
                            first_half_embeddings = batch_embeddings[:half_size]
                            first_half_metadata = batch_metadata[:half_size] if batch_metadata else None
                            
                            inserted1 = self._insert_single_batch(
                                first_half_chunks, first_half_embeddings, document_id, file_hash, file_name,
                                file_size, technology, domain, first_half_metadata, file_id,
                                batch_chunk_ids[:half_size] if batch_chunk_ids else None,
                                start_chunk_index=i, start_char_pos=global_char_pos, total_chunks_in_doc=total_chunks,
                                chunk_indices=batch_chunk_indices[:half_size] if batch_chunk_indices else None,
                                chunk_char_starts=batch_char_starts[:half_size] if batch_char_starts else None,
                            )
                            total_inserted += inserted1
                            
                            # Update position
                            for chunk in first_half_chunks:
                                global_char_pos += len(chunk)
                            
                            # Retry second half
                            second_half_chunks = batch_chunks[half_size:]
                            second_half_embeddings = batch_embeddings[half_size:]
                            second_half_metadata = batch_metadata[half_size:] if batch_metadata else None
                            
                            inserted2 = self._insert_single_batch(
                                second_half_chunks, second_half_embeddings, document_id, file_hash, file_name,
                                file_size, technology, domain, second_half_metadata, file_id,
                                batch_chunk_ids[half_size:] if batch_chunk_ids else None,
                                start_chunk_index=i + half_size, start_char_pos=global_char_pos, total_chunks_in_doc=total_chunks,
                                chunk_indices=batch_chunk_indices[half_size:] if batch_chunk_indices else None,
                                chunk_char_starts=batch_char_starts[half_size:] if batch_char_starts else None,
                            )
                            total_inserted += inserted2
                            
                            # Update position
                            for chunk in second_half_chunks:
                                global_char_pos += len(chunk)
                            
                            print(f">>> VECTOR STORE: Successfully inserted batch {batch_num} in 2 smaller batches", flush=True)
                            continue  # Successfully handled, continue to next batch
                            
                        except Exception as retry_error:
                            print(f">>> VECTOR STORE: Retry with smaller batches also failed: {str(retry_error)}", flush=True)
                            raise
                    else:
                        print(f">>> VECTOR STORE: Cannot split batch further. Batch size: {len(batch_chunks)}", flush=True)
                        raise
                else:
                    # Other errors - just raise
                    print(f">>> VECTOR STORE ERROR in batch {batch_num}/{total_batches}: {error_msg}", flush=True)
                    raise
        
        print(f">>> VECTOR STORE: Successfully inserted {total_inserted} chunks", flush=True)
        logger.info(f"Inserted {total_inserted} chunks into collection '{self.collection_name}' (technology={technology}, domain={domain})")
        return total_inserted
    
    def _insert_single_batch(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        document_id: str,
        file_hash: str,
        file_name: str,
        file_size: int,
        technology: str,
        domain: str,
        chunk_metadata: Optional[List[Dict]],
        file_id: Optional[str],
        chunk_ids: Optional[List[str]] = None,
        start_chunk_index: int = 0,
        start_char_pos: int = 0,
        total_chunks_in_doc: Optional[int] = None,
        chunk_indices: Optional[List[int]] = None,
        chunk_char_starts: Optional[List[int]] = None,
    ) -> int:
        """
        Helper method to insert a single batch of chunks.
        
        Args:
            chunks: List of text chunks for this batch
            embeddings: List of embedding vectors for this batch
            document_id: Unique document identifier
            file_hash: SHA256 hash of the file
            file_name: Original filename
            file_size: Size of the file in bytes
            technology: Technology category
            domain: Domain category
            chunk_metadata: Optional list of metadata dicts per chunk
            file_id: Optional file identifier
            start_chunk_index: Starting chunk index (for multi-batch documents)
            start_char_pos: Starting character position (for multi-batch documents)
            total_chunks_in_doc: Total chunks in the entire document
            
        Returns:
            Number of chunks inserted
        """
        now_ts = int(time.time())
        total_chunks = total_chunks_in_doc if total_chunks_in_doc is not None else len(chunks)

        # Prepare data for insertion
        data = []
        current_pos = start_char_pos
        explicit_idx = chunk_indices is not None and len(chunk_indices) == len(chunks)
        explicit_char = chunk_char_starts is not None and len(chunk_char_starts) == len(chunks)
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if explicit_idx:
                chunk_index = chunk_indices[i]
            else:
                chunk_index = start_chunk_index + i
            if explicit_char:
                start_char = chunk_char_starts[i]
                end_char = start_char + len(chunk)
            else:
                start_char = current_pos
                end_char = current_pos + len(chunk)
                current_pos = end_char

            meta = chunk_metadata[i] if chunk_metadata and i < len(chunk_metadata) else {}
            chunk_id = ""
            if chunk_ids and i < len(chunk_ids):
                chunk_id = chunk_ids[i]
            elif isinstance(meta, dict):
                chunk_id = meta.get("chunk_id", "")

            data.append({
                "vector": embedding,
                "text": chunk[:self.MAX_TEXT_LEN],
                "text_length": len(chunk),
                "document_id": document_id,
                "file_name": file_name,
                "file_hash": file_hash,
                "file_id": file_id if file_id else "",
                "chunk_id": chunk_id,
                "technology": technology[:50],
                "domain": domain[:50],
                "chunk_index": chunk_index,
                "chunk_start_char": start_char,
                "chunk_end_char": end_char,
                "total_chunks_in_doc": total_chunks,
                "created_at": now_ts,
                "updated_at": now_ts,
                "page_number": meta.get("page_number", -1),
                "section_title": meta.get("section_title", "")[:500],
                "chunk_type": meta.get("chunk_type", "paragraph")[:50],
                "file_size": file_size,
                "file_type": meta.get("file_type", "pdf")[:20],
                "language": meta.get("language", "en")[:10],
                "has_tables": 1 if meta.get("has_tables", False) else 0,
                "has_images": 1 if meta.get("has_images", False) else 0,
                "metadata_json": json.dumps(meta)[:10000],
            })

        # Insert using REST API: /v2/vectordb/entities/insert
        insert_payload = {
            "collectionName": self.collection_name,
            "data": data,
        }

        response = httpx.post(
            self._get_api_url("insert"),
            headers=self.headers,
            json=insert_payload,
            timeout=_zilliz_insert_timeout(),
        )
        response.raise_for_status()
        
        return len(data)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        technology: Optional[str] = None,
        domain: Optional[str] = None,
        document_id: Optional[str] = None,
        file_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Search for similar chunks in the vector store.
        
        Args:
            query_embedding: Query vector embedding
            top_k: Number of results to return
            technology: Filter by technology (e.g., 'react', 'python')
            domain: Filter by domain (e.g., 'frontend', 'backend')
            document_id: Filter by specific document ID
            file_id: Filter by specific file ID
            document_ids: Filter to any of these document IDs (scoped chat pool)
            
        Returns:
            List of matching chunks with metadata
        """
        expr = _build_search_filter_expr(
            technology=technology,
            domain=domain,
            document_id=document_id,
            file_id=file_id,
            document_ids=document_ids,
        )
        search_payload = _build_search_payload(
            self.collection_name, query_embedding, top_k, expr
        )
        logger.info(f"Search request: collection={self.collection_name}, limit={top_k}, filter={expr}")

        response = httpx.post(
            self._get_api_url("search"),
            headers=self.headers,
            json=search_payload,
            timeout=30.0,
        )
        result_data = response.json()
        logger.info(
            f"Search response: code={result_data.get('code')}, message={result_data.get('message', 'N/A')}"
        )
        formatted = _format_zilliz_search_hits(result_data)
        logger.info(f"Search returned {len(formatted)} results")
        return formatted

    async def search_async(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        technology: Optional[str] = None,
        domain: Optional[str] = None,
        document_id: Optional[str] = None,
        file_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> List[Dict]:
        """Async vector search (shared httpx client when provided)."""
        expr = _build_search_filter_expr(
            technology=technology,
            domain=domain,
            document_id=document_id,
            file_id=file_id,
            document_ids=document_ids,
        )
        search_payload = _build_search_payload(
            self.collection_name, query_embedding, top_k, expr
        )
        url = self._get_api_url("search")
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=30.0)
        try:
            response = await client.post(url, headers=self.headers, json=search_payload)
            result_data = response.json()
            return _format_zilliz_search_hits(result_data)
        finally:
            if owns_client:
                await client.aclose()

    async def delete_by_document_id_async(self, document_id: str, client: httpx.AsyncClient) -> bool:
        """Delete entities by document_id (async HTTP)."""
        if not document_id or not str(document_id).strip():
            logger.warning("delete_by_document_id: skipped empty document_id")
            return False
        delete_payload = {
            "collectionName": self.collection_name,
            "filter": f'document_id == "{document_id}"',
        }
        response = await client.post(
            self._get_api_url("delete"),
            headers=self.headers,
            json=delete_payload,
            timeout=_zilliz_delete_timeout(),
        )
        response.raise_for_status()
        logger.info(
            "Deleted chunks async document_id=%s collection=%s",
            document_id[:8],
            self.collection_name,
        )
        return True

    def delete_by_document_id(self, document_id: str) -> bool:
        """Sync wrapper for non-async callers (e.g. ingest rollback)."""
        return asyncio.run(self._delete_by_document_id_async_single(document_id))

    async def _delete_by_document_id_async_single(self, document_id: str) -> bool:
        timeout = httpx.Timeout(_zilliz_delete_timeout())
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await self.delete_by_document_id_async(document_id, client)

    def delete_by_file_id(self, file_id: str) -> bool:
        """Delete entities by file_id using REST API."""
        if not file_id or not str(file_id).strip():
            logger.warning("delete_by_file_id: skipped empty file_id")
            return False
        delete_payload = {
            "collectionName": self.collection_name,
            "filter": f'file_id == "{file_id}"',
        }

        response = httpx.post(
            self._get_api_url("delete"),
            headers=self.headers,
            json=delete_payload,
            timeout=30.0,
        )
        response.raise_for_status()
        logger.info(f"Deleted chunks for file_id={file_id}")
        return True

    def delete_by_technology(self, technology: str) -> bool:
        """Delete all entities for a specific technology."""
        delete_payload = {
            "collectionName": self.collection_name,
            "filter": f'technology == "{technology}"',
        }

        response = httpx.post(
            self._get_api_url("delete"),
            headers=self.headers,
            json=delete_payload,
            timeout=30.0,
        )
        response.raise_for_status()
        logger.info(f"Deleted chunks for technology={technology}")
        return True


class VectorStore:
    """
    Routes chunks to separate Zilliz collections: text (incl. image captions), code, tables.
    Search runs per collection with query-aware top_k allocation and score boosting, then merges.
    """

    def __init__(self, uri: str, token: str, collection_name: str):
        base = (collection_name or "pdf_chatbot_collection").strip()
        self.base_collection_name = base
        self._by_bucket = {
            "text": _ZillizCollection(uri, token, f"{base}_{TEXT_SUFFIX}"),
            "code": _ZillizCollection(uri, token, f"{base}_{CODE_SUFFIX}"),
            "tables": _ZillizCollection(uri, token, f"{base}_{TABLES_SUFFIX}"),
        }

    def insert_chunks(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        document_id: str,
        file_hash: str,
        file_name: str,
        file_size: int,
        technology: str = "general",
        domain: str = "general",
        chunk_metadata: Optional[List[Dict]] = None,
        file_id: Optional[str] = None,
        chunk_ids: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        chunk_indices: Optional[List[int]] = None,
        chunk_char_starts: Optional[List[int]] = None,
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings length mismatch")
        if not chunks:
            return 0
        if chunk_metadata is not None and len(chunk_metadata) != len(chunks):
            raise ValueError("Chunks and chunk_metadata length mismatch")
        if chunk_ids and len(chunk_ids) != len(chunks):
            raise ValueError("Chunks and chunk_ids length mismatch")

        n = len(chunks)
        idx_list = list(chunk_indices) if chunk_indices is not None else list(range(n))
        if chunk_char_starts is not None:
            if len(chunk_char_starts) != n:
                raise ValueError("chunk_char_starts length mismatch")
            char_list = list(chunk_char_starts)
        else:
            pos = 0
            char_list = []
            for c in chunks:
                char_list.append(pos)
                pos += len(c)

        per_bucket: Dict[str, List[int]] = {k: [] for k in _BUCKET_ORDER}
        for i in range(n):
            meta = chunk_metadata[i] if chunk_metadata else {}
            b = chunk_type_to_bucket(meta.get("chunk_type"))
            per_bucket[b].append(i)

        total_inserted = 0
        try:
            for bucket in _BUCKET_ORDER:
                idxs = per_bucket[bucket]
                if not idxs:
                    continue
                store = self._by_bucket[bucket]
                sub_chunks = [chunks[i] for i in idxs]
                sub_emb = [embeddings[i] for i in idxs]
                sub_meta = [chunk_metadata[i] for i in idxs] if chunk_metadata else None
                sub_ids = [chunk_ids[i] for i in idxs] if chunk_ids else None
                sub_idx = [idx_list[i] for i in idxs]
                sub_char = [char_list[i] for i in idxs]
                total_inserted += store.insert_chunks(
                    sub_chunks,
                    sub_emb,
                    document_id,
                    file_hash,
                    file_name,
                    file_size,
                    technology=technology,
                    domain=domain,
                    chunk_metadata=sub_meta,
                    file_id=file_id,
                    chunk_ids=sub_ids,
                    batch_size=batch_size,
                    chunk_indices=sub_idx,
                    chunk_char_starts=sub_char,
                )
        except Exception:
            logger.warning(
                "Typed vector insert failed for document_id=%s; rolling back all buckets for this doc",
                document_id,
            )
            try:
                self.delete_by_document_id(document_id)
            except Exception as rb_err:
                logger.warning("VectorStore rollback delete_by_document_id failed: %s", rb_err)
            raise
        logger.info(
            "Typed insert into Zilliz base=%s total_rows=%s (text+images / code / tables routed)",
            self.base_collection_name,
            total_inserted,
        )
        return total_inserted

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        technology: Optional[str] = None,
        domain: Optional[str] = None,
        document_id: Optional[str] = None,
        file_id: Optional[str] = None,
        query_text: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Search every typed collection (text, code, tables), merge, re-score by query intent, return top_k.
        Chat retrieval always goes through this path — not a single-collection shortcut.
        """
        weights = infer_retrieval_bucket_weights(query_text or "")
        limits = _allocate_bucket_top_k(top_k, weights)
        wmax = max(weights.values()) or 1.0

        merged: List[Dict] = []
        for bucket in _BUCKET_ORDER:
            lim = limits[bucket]
            if lim <= 0:
                continue
            hits = self._by_bucket[bucket].search(
                query_embedding=query_embedding,
                top_k=lim,
                technology=technology,
                domain=domain,
                document_id=document_id,
                file_id=file_id,
                document_ids=document_ids,
            )
            wb = weights.get(bucket, 0.0)
            for h in hits:
                raw = h.get("score", 0.0)
                sim = _normalize_similarity_score(raw)
                h["raw_vector_score"] = raw
                h["vector_bucket"] = bucket
                h["score"] = sim * (0.55 + 0.45 * (wb / wmax))
                merged.append(h)

        merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return merged[:top_k]

    async def search_async(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        technology: Optional[str] = None,
        domain: Optional[str] = None,
        document_id: Optional[str] = None,
        file_id: Optional[str] = None,
        query_text: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Search text/code/table collections concurrently, merge, re-score."""
        weights = infer_retrieval_bucket_weights(query_text or "")
        limits = _allocate_bucket_top_k(top_k, weights)
        wmax = max(weights.values()) or 1.0

        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = []
            bucket_keys = []
            for bucket in _BUCKET_ORDER:
                lim = limits[bucket]
                if lim <= 0:
                    continue
                bucket_keys.append(bucket)
                tasks.append(
                    self._by_bucket[bucket].search_async(
                        query_embedding=query_embedding,
                        top_k=lim,
                        technology=technology,
                        domain=domain,
                        document_id=document_id,
                        file_id=file_id,
                        document_ids=document_ids,
                        client=client,
                    )
                )
            if not tasks:
                return []
            results = await asyncio.gather(*tasks)

        merged: List[Dict] = []
        for bucket, hits in zip(bucket_keys, results):
            wb = weights.get(bucket, 0.0)
            for h in hits:
                raw = h.get("score", 0.0)
                sim = _normalize_similarity_score(raw)
                h["raw_vector_score"] = raw
                h["vector_bucket"] = bucket
                h["score"] = sim * (0.55 + 0.45 * (wb / wmax))
                merged.append(h)

        merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return merged[:top_k]

    async def insert_chunks_async(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        document_id: str,
        file_hash: str,
        file_name: str,
        file_size: int,
        technology: str = "general",
        domain: str = "general",
        chunk_metadata: Optional[List[Dict]] = None,
        file_id: Optional[str] = None,
        chunk_ids: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        chunk_indices: Optional[List[int]] = None,
        chunk_char_starts: Optional[List[int]] = None,
    ) -> int:
        """Insert into typed buckets in parallel (thread pool per bucket HTTP batches)."""
        if not chunks:
            return 0

        def _insert_bucket(bucket: str) -> int:
            if len(chunks) != len(embeddings):
                raise ValueError("Chunks and embeddings length mismatch")
            n = len(chunks)
            idx_list = list(chunk_indices) if chunk_indices is not None else list(range(n))
            if chunk_char_starts is not None:
                if len(chunk_char_starts) != n:
                    raise ValueError("chunk_char_starts length mismatch")
                char_list = list(chunk_char_starts)
            else:
                pos = 0
                char_list = []
                for c in chunks:
                    char_list.append(pos)
                    pos += len(c)
            per_bucket: Dict[str, List[int]] = {k: [] for k in _BUCKET_ORDER}
            for i in range(n):
                meta = chunk_metadata[i] if chunk_metadata else {}
                b = chunk_type_to_bucket(meta.get("chunk_type"))
                per_bucket[b].append(i)
            idxs = per_bucket[bucket]
            if not idxs:
                return 0
            store = self._by_bucket[bucket]
            sub_chunks = [chunks[i] for i in idxs]
            sub_emb = [embeddings[i] for i in idxs]
            sub_meta = [chunk_metadata[i] for i in idxs] if chunk_metadata else None
            sub_ids = [chunk_ids[i] for i in idxs] if chunk_ids else None
            sub_idx = [idx_list[i] for i in idxs]
            sub_char = [char_list[i] for i in idxs]
            return store.insert_chunks(
                sub_chunks,
                sub_emb,
                document_id,
                file_hash,
                file_name,
                file_size,
                technology=technology,
                domain=domain,
                chunk_metadata=sub_meta,
                file_id=file_id,
                chunk_ids=sub_ids,
                batch_size=batch_size,
                chunk_indices=sub_idx,
                chunk_char_starts=sub_char,
            )

        try:
            results = await asyncio.gather(
                *[
                    asyncio.to_thread(_insert_bucket, bucket)
                    for bucket in _BUCKET_ORDER
                ],
                return_exceptions=True,
            )
        except Exception:
            try:
                await self.delete_by_document_id_async(document_id)
            except Exception as rb_err:
                logger.warning("VectorStore async insert rollback failed: %s", rb_err)
            raise

        total_inserted = 0
        for bucket, result in zip(_BUCKET_ORDER, results):
            if isinstance(result, Exception):
                logger.warning(
                    "Typed async insert failed bucket=%s doc=%s: %s",
                    bucket,
                    document_id[:8],
                    result,
                )
                try:
                    await self.delete_by_document_id_async(document_id)
                except Exception as rb_err:
                    logger.warning("VectorStore async insert rollback failed: %s", rb_err)
                raise result
            total_inserted += int(result or 0)

        logger.info(
            "Typed async insert into Zilliz base=%s total_rows=%s",
            self.base_collection_name,
            total_inserted,
        )
        return total_inserted

    async def delete_by_document_id_async(self, document_id: str) -> bool:
        """Delete from text/code/table collections concurrently via async HTTP."""
        if not document_id or not str(document_id).strip():
            return False
        stores = list(self._by_bucket.values())
        if not stores:
            return False
        if len(stores) == 1:
            return await stores[0]._delete_by_document_id_async_single(document_id)

        timeout = httpx.Timeout(_zilliz_delete_timeout())
        async with httpx.AsyncClient(timeout=timeout) as client:
            results = await asyncio.gather(
                *[s.delete_by_document_id_async(document_id, client) for s in stores],
                return_exceptions=True,
            )
        ok_any = False
        for store, result in zip(stores, results):
            if isinstance(result, Exception):
                logger.warning(
                    "Zilliz async delete failed collection=%s doc=%s: %s",
                    store.collection_name,
                    document_id[:8],
                    result,
                )
            elif result:
                ok_any = True
        return ok_any

    def delete_by_document_id(self, document_id: str) -> bool:
        """Sync wrapper for non-async callers."""
        return asyncio.run(self.delete_by_document_id_async(document_id))

    def delete_by_file_id(self, file_id: str) -> bool:
        ok_any = False
        for store in self._by_bucket.values():
            if store.delete_by_file_id(file_id):
                ok_any = True
        return ok_any

    def delete_by_technology(self, technology: str) -> bool:
        ok_any = False
        for store in self._by_bucket.values():
            if store.delete_by_technology(technology):
                ok_any = True
        return ok_any
