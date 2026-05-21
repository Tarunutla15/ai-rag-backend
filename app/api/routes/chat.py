"""Chat API route - RAG Q&A with session and context."""
import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
from app.config import settings, get_completion_client_config
from app.models.schemas import ChatRequest, ChatResponse
from app.services.chat_service import get_chat_service
from app.services.query_classifier import QueryClassifier, classify_query_with_context, classify_query
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.document_store import get_document_store as _get_document_store
from app.services.reranker import RerankerService
from app.services.keyword_search import get_keyword_search_service
from app.services.chunk_store import get_chunk_store
from app.services.raw_block_store import get_raw_block_store
from app.services.usage_store import record_chat_completion
import json

# Reuse vector store getter from upload route to avoid duplicate init logic
from app.api.routes.upload import get_vector_store

router = APIRouter(prefix="/chat", tags=["chat"])

# Lazy-initialized services
_embedding_service = None
_llm_service = None
_document_service = None
_reranker_service = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_EMBEDDING_MODEL,
        )
    return _embedding_service


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        key, model, base_url = get_completion_client_config()
        _llm_service = LLMService(api_key=key, model=model, base_url=base_url)
    return _llm_service


def get_document_store():
    """Get DB-backed document store singleton."""
    return _get_document_store()


def get_reranker_service() -> RerankerService:
    global _reranker_service
    if _reranker_service is None:
        key, model, base_url = get_completion_client_config()
        _reranker_service = RerankerService(api_key=key, model=model, base_url=base_url)
    return _reranker_service


def _get_previous_user_message(recent_messages: list) -> str:
    """Return the most recent user message before the current one, if any."""
    if not recent_messages or len(recent_messages) < 2:
        return ""
    # recent_messages are chronological; last item is current user query
    for msg in reversed(recent_messages[:-1]):
        if msg.get("role") == "user" and (msg.get("content") or "").strip():
            return (msg.get("content") or "").strip()
    return ""


def _technologies_from_scoped_documents(scope_file_ids: list) -> tuple:
    """
    Derive retrieval technology/domain from ingested document metadata when the user
    scoped the chat to specific PDFs. Avoids LLM returning java,python,js for generic
    questions like "what is a function" while only Python_Notes.pdf is in scope.
    """
    if not scope_file_ids:
        return [], None
    store = get_document_store()
    techs: list = []
    domains: list = []
    for did in scope_file_ids:
        doc = store.get_document(did)
        if not doc:
            continue
        t = (doc.get("technology") or "").strip().lower()
        d = (doc.get("domain") or "").strip().lower()
        if t and t not in techs:
            techs.append(t)
        if d and d not in domains:
            domains.append(d)
    if not techs:
        return [], None
    domain = domains[0] if len(domains) == 1 else None
    if len(techs) == 1 and not domain:
        domain = QueryClassifier.TECHNOLOGY_TO_DOMAIN.get(techs[0])
    return techs, domain


def _get_last_user_message_with_tech(recent_messages: list) -> tuple:
    """
    Return (text, technologies, domain) for the most recent user message that yields a tech.
    Uses classify_query on user messages only.
    """
    if not recent_messages or len(recent_messages) < 2:
        return "", [], None
    for msg in reversed(recent_messages[:-1]):
        if msg.get("role") != "user":
            continue
        text = (msg.get("content") or "").strip()
        if not text:
            continue
        techs, dom = classify_query(text)
        if techs:
            return text, techs, dom
    return "", [], None


_FOLLOWUP_PRONOUN_RE = re.compile(
    r"\b(it|this|that|those|them|same|more|above|earlier|previous)\b",
    re.IGNORECASE,
)
_QUERY_STOPWORDS = frozenset({
    "what", "are", "is", "was", "the", "a", "an", "and", "or", "give", "some", "how",
    "to", "in", "of", "for", "with", "about", "explain", "tell", "me", "can", "you",
    "depth", "examples", "example", "please",
    # Filler adjectives / common typos (e.g. "nad" for "and") — not useful for ILIKE or section boost
    "good", "best", "great", "nice", "very", "really", "just", "also", "nad", "adn", "nd",
})
_TECH_NAME_STOPWORDS = frozenset(QueryClassifier.TECHNOLOGY_TO_DOMAIN.keys())


def _is_short_follow_up(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    if len(q) <= 48 and _FOLLOWUP_PRONOUN_RE.search(q):
        return True
    return len(q) <= 28


def _topic_terms(query: str, technologies: Optional[List[str]] = None) -> list:
    """Content terms for section boost / keyword focus (exclude tech names and stopwords)."""
    exclude = set(_QUERY_STOPWORDS)
    exclude |= _TECH_NAME_STOPWORDS
    if technologies:
        exclude |= {t.lower() for t in technologies}
    tokens = re.findall(r"[A-Za-z0-9_]+", query or "")
    out: list = []
    for t in tokens:
        low = t.lower()
        if len(low) < 3 or low in exclude:
            continue
        if low.endswith("s") and len(low) > 4 and low[:-1] not in exclude:
            out.append(low[:-1])
        out.append(low)
    seen: set = set()
    deduped: list = []
    for t in out:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped[:8]


def _build_search_query(
    query: str,
    technologies: list,
    from_conversation_context: bool,
    recent_messages: list,
    topic_context_text: str = "",
    *,
    from_scoped_documents: bool = False,
) -> str:
    """
    Build the string used for vector embeddings.

    - Scoped PDFs: do not prepend technology (document_id filter already applies).
    - Conversation-inferred scope: prepend technology slugs to bias embedding.
    - Short pronoun follow-ups: stitch prior user turn so "it" resolves to the topic.
    """
    base = query.strip()
    if from_conversation_context or from_scoped_documents:
        if _is_short_follow_up(base):
            prev_user = topic_context_text or _get_previous_user_message(recent_messages)
            if prev_user:
                base = f"{prev_user} {base}".strip()
    if from_conversation_context and technologies and not from_scoped_documents:
        return (" ".join(technologies) + " " + base).strip()
    return base


def _build_keyword_query(
    query: str,
    recent_messages: list,
    technologies: list | None,
    *,
    from_scoped_documents: bool = False,
) -> str:
    """Keyword ILIKE tokens — never prepend technology when the PDF is already scoped."""
    base = query.strip()
    if _is_short_follow_up(base):
        prev_user = _get_previous_user_message(recent_messages)
        if prev_user:
            base = f"{prev_user} {base}".strip()
    return base


def _build_rerank_query(query: str, recent_messages: list) -> str:
    """Reranker should see the full topic on pronoun follow-ups."""
    base = query.strip()
    if _is_short_follow_up(base):
        prev_user = _get_previous_user_message(recent_messages)
        if prev_user:
            return f"{prev_user} {base}".strip()
    return base


def _looks_like_toc_excerpt(text: str) -> bool:
    t = (text or "")[:2500]
    if "Introduction to Python" in t and ("What is Python?" in t or "Download and Install" in t):
        return True
    dotted = len(re.findall(r"\b\w+\s+\d{1,3}\s*$", t, re.MULTILINE))
    return dotted >= 6


def _apply_retrieval_relevance_tuning(chunks: list, topic_terms: list) -> list:
    """Boost on-topic sections; down-rank table-of-contents style excerpts."""
    if not chunks:
        return chunks
    for c in chunks:
        score = float(c.get("hybrid_score", c.get("score", 0.0)) or 0.0)
        title = (c.get("section_title") or "").lower()
        text = (c.get("text") or "").lower()
        if topic_terms:
            hits = sum(
                1 for term in topic_terms
                if term in title or term in text or f"{term}s" in text
            )
            if hits:
                score += 0.14 * hits
        chunk_type = (c.get("chunk_type") or "").lower()
        if chunk_type == "heading" and topic_terms:
            heading_boost = float(getattr(settings, "HEADING_RETRIEVAL_BOOST", 0.12))
            if any(
                term in title or term in text or f"{term}s" in text
                for term in topic_terms
            ):
                score += heading_boost
        metadata_json = c.get("metadata_json") or ""
        if topic_terms and metadata_json:
            meta_lower = metadata_json.lower() if isinstance(metadata_json, str) else ""
            path_hits = sum(1 for term in topic_terms if term in meta_lower)
            if path_hits:
                score += 0.06 * path_hits
        if _looks_like_toc_excerpt(c.get("text") or ""):
            score *= 0.45
        c["hybrid_score"] = score
        c["score"] = score
    chunks.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    return chunks


def _cap_scope_document_ids(scope_file_ids: list) -> list:
    """Limit in-filter document count for Zilliz/Supabase."""
    ids = [str(d).strip() for d in (scope_file_ids or []) if d and str(d).strip()]
    max_docs = int(getattr(settings, "CHAT_SCOPED_POOL_MAX_DOCUMENTS", 40))
    if len(ids) > max_docs:
        logger.warning(
            ">>> RETRIEVAL: scope has %s documents; capping pooled search to %s",
            len(ids),
            max_docs,
        )
        return ids[:max_docs]
    return ids


async def _retrieve_scoped_document_pool_async(
    *,
    search_query: str,
    keyword_query: str,
    rerank_query: str,
    scope_file_ids: list,
    vector_store,
    embedding_service: EmbeddingService,
    keyword_service,
    enable_keyword: bool,
    keyword_top_k: int,
    vector_weight: float,
    keyword_weight: float,
    enable_reranker: bool,
    initial_top_k: int,
    rerank_top_n: int,
    topic_terms: Optional[List[str]] = None,
) -> list:
    """
    One hybrid retrieval pass over the user's scoped document IDs (async I/O).

    Embedding runs first; vector (3 Zilliz buckets in parallel) and keyword FTS run concurrently.
    """
    doc_ids = _cap_scope_document_ids(scope_file_ids)
    if not doc_ids:
        return []

    query_embedding = await embedding_service.generate_embedding_async(search_query)
    logger.info(
        ">>> RETRIEVAL (async): scoped pool search docs=%s top_k=%s",
        len(doc_ids),
        initial_top_k,
    )

    vector_coro = vector_store.search_async(
        query_embedding=query_embedding,
        top_k=initial_top_k,
        technology=None,
        domain=None,
        document_ids=doc_ids,
        query_text=search_query,
    )
    if enable_keyword and keyword_service:
        keyword_coro = asyncio.to_thread(
            keyword_service.search,
            query=keyword_query,
            top_k=keyword_top_k,
            technology=None,
            domain=None,
            document_ids=doc_ids,
        )
        vector_results, keyword_results = await asyncio.gather(vector_coro, keyword_coro)
    else:
        vector_results = await vector_coro
        keyword_results = []

    merged = _merge_hybrid_results(vector_results, keyword_results, vector_weight, keyword_weight)
    merged = _apply_retrieval_relevance_tuning(merged, topic_terms or [])

    if not merged:
        logger.info(">>> RETRIEVAL: scoped pool returned 0 chunks for docs=%s", len(doc_ids))
        return []

    contributing = len({c.get("document_id") for c in merged if c.get("document_id")})
    logger.info(
        ">>> RETRIEVAL: scoped pool vector=%s keyword=%s merged=%s contributing_docs=%s/%s",
        len(vector_results),
        len(keyword_results),
        len(merged),
        contributing,
        len(doc_ids),
    )

    if enable_reranker and len(merged) > rerank_top_n:
        logger.info(">>> RERANK: scoped pool input_chunks=%s -> top_n=%s", len(merged), rerank_top_n)
        reranker = get_reranker_service()
        merged = await asyncio.to_thread(
            reranker.rerank,
            query=rerank_query,
            chunks=merged,
            top_n=rerank_top_n,
        )
        logger.info(">>> RERANK: scoped pool output_chunks=%s", len(merged))
    elif len(merged) > rerank_top_n:
        merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        merged = merged[:rerank_top_n]

    return merged


def _retrieve_context_chunks(
    *,
    search_query: str,
    keyword_query: str,
    rerank_query: str,
    technologies: list,
    domain: str,
    request: ChatRequest,
    vector_store,
    embedding_service: EmbeddingService,
    keyword_service,
    enable_keyword: bool,
    keyword_top_k: int,
    vector_weight: float,
    keyword_weight: float,
    enable_reranker: bool,
    initial_top_k: int,
    rerank_top_n: int,
    max_chunks_no_rerank: int,
    topic_terms: Optional[List[str]] = None,
) -> list:
    """
    Retrieve and (optionally) rerank context chunks.

    Vector search uses ``VectorStore.search`` (``vector_store.py``): that queries **all three**
    Zilliz collections (``{base}_text_chunks``, ``{base}_code_blocks``, ``{base}_tables``),
    merges results, then returns the top-k combined hits — not a single flat collection.
    """
    query_embedding = embedding_service.generate_embedding(search_query)

    # Balanced retrieval when scope has 2+ technologies
    if len(technologies) >= 2:
        logger.info(">>> RETRIEVAL: multi-scope balanced retrieval for technologies=%s", technologies)
        context_chunks = []
        reranker = get_reranker_service() if enable_reranker else None
        per_tech_top_n = max(1, rerank_top_n)
        scoped_doc = request.file_id

        def _hybrid_merged(tech: str, dom: Optional[str]) -> tuple[list, list, list]:
            vr = vector_store.search(
                query_embedding=query_embedding,
                top_k=initial_top_k,
                technology=tech,
                domain=dom,
                document_id=request.file_id,
                file_id=request.file_id,
                query_text=search_query,
            )
            kr = []
            if enable_keyword and keyword_service:
                kr = keyword_service.search(
                    query=keyword_query,
                    top_k=keyword_top_k,
                    technology=tech,
                    domain=dom,
                    document_id=request.file_id,
                    file_id=request.file_id,
                )
            merged = _merge_hybrid_results(vr, kr, vector_weight, keyword_weight)
            return vr, kr, merged

        for tech in technologies:
            tech_domain = QueryClassifier.TECHNOLOGY_TO_DOMAIN.get(tech, "general")
            vector_results, keyword_results, merged = _hybrid_merged(tech, tech_domain)
            if not merged and scoped_doc:
                vector_results, keyword_results, merged = _hybrid_merged(tech, None)
                if merged:
                    logger.info(
                        ">>> RETRIEVAL: technology=%s using domain-agnostic filter (ingest domain != query routing)",
                        tech,
                    )
            logger.info(
                ">>> RETRIEVAL: technology=%s vector=%s keyword=%s merged=%s",
                tech, len(vector_results), len(keyword_results), len(merged)
            )
            if not merged:
                continue
            if enable_reranker and len(merged) > per_tech_top_n and reranker:
                kept = reranker.rerank(query=rerank_query, chunks=merged, top_n=per_tech_top_n)
            else:
                kept = merged[:per_tech_top_n]
            context_chunks.extend(kept)
        # Chunks are stored with document-level tech/domain; query routing may use java+backend while ingest used general
        if not context_chunks and scoped_doc:
            wide_top = max(initial_top_k * 2, rerank_top_n * 4)
            logger.info(
                ">>> RETRIEVAL: multi-tech still empty; document-wide search document_id=%s (no tech/domain filter)",
                scoped_doc,
            )
            vr_w = vector_store.search(
                query_embedding=query_embedding,
                top_k=wide_top,
                technology=None,
                domain=None,
                document_id=scoped_doc,
                file_id=scoped_doc,
                query_text=search_query,
            )
            kr_w = []
            if enable_keyword and keyword_service:
                kr_w = keyword_service.search(
                    query=keyword_query,
                    top_k=max(keyword_top_k, wide_top),
                    technology=None,
                    domain=None,
                    document_id=scoped_doc,
                    file_id=scoped_doc,
                )
            merged_wide = _merge_hybrid_results(vr_w, kr_w, vector_weight, keyword_weight)
            logger.info(
                ">>> RETRIEVAL: document-wide vector=%s keyword=%s merged=%s",
                len(vr_w), len(kr_w), len(merged_wide),
            )
            if merged_wide:
                if enable_reranker and len(merged_wide) > rerank_top_n and reranker:
                    context_chunks = reranker.rerank(query=rerank_query, chunks=merged_wide, top_n=rerank_top_n)
                elif len(merged_wide) > rerank_top_n:
                    merged_wide.sort(key=lambda x: x.get("score", 0.0), reverse=True)
                    context_chunks = merged_wide[:rerank_top_n]
                else:
                    context_chunks = merged_wide
        context_chunks = _apply_retrieval_relevance_tuning(context_chunks, topic_terms or [])
        # One global rerank across technologies so ordering is not biased by tech loop order
        if enable_reranker and reranker and len(context_chunks) > rerank_top_n:
            context_chunks = reranker.rerank(
                query=rerank_query, chunks=context_chunks, top_n=rerank_top_n
            )
        elif len(context_chunks) > rerank_top_n:
            context_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            context_chunks = context_chunks[:rerank_top_n]
        logger.info(">>> RETRIEVAL: balanced total context_chunks=%s (sent to LLM)", len(context_chunks))
        return context_chunks

    # Single scope or no scope: one search, then rerank
    technology = technologies[0] if technologies else None
    logger.info(
        ">>> RETRIEVAL: vector_search top_k=%s (reranker_enabled=%s, rerank_keep=%s)",
        initial_top_k if enable_reranker else max_chunks_no_rerank,
        enable_reranker,
        rerank_top_n if enable_reranker else "n/a",
    )
    scoped_doc = request.file_id
    vector_results = vector_store.search(
        query_embedding=query_embedding,
        top_k=initial_top_k if enable_reranker else max_chunks_no_rerank,
        technology=technology,
        domain=domain,
        document_id=request.file_id,
        file_id=request.file_id,
        query_text=search_query,
    )
    keyword_results = []
    if enable_keyword and keyword_service:
        keyword_results = keyword_service.search(
            query=keyword_query,
            top_k=keyword_top_k,
            technology=technology,
            domain=domain,
            document_id=request.file_id,
            file_id=request.file_id,
        )
    merged = _merge_hybrid_results(vector_results, keyword_results, vector_weight, keyword_weight)
    merged = _apply_retrieval_relevance_tuning(merged, topic_terms or [])
    if not merged and scoped_doc and domain:
        vector_results = vector_store.search(
            query_embedding=query_embedding,
            top_k=initial_top_k if enable_reranker else max_chunks_no_rerank,
            technology=technology,
            domain=None,
            document_id=request.file_id,
            file_id=request.file_id,
            query_text=search_query,
        )
        keyword_results = []
        if enable_keyword and keyword_service:
            keyword_results = keyword_service.search(
                query=keyword_query,
                top_k=keyword_top_k,
                technology=technology,
                domain=None,
                document_id=request.file_id,
                file_id=request.file_id,
            )
        merged = _merge_hybrid_results(vector_results, keyword_results, vector_weight, keyword_weight)
        merged = _apply_retrieval_relevance_tuning(merged, topic_terms or [])
        if merged:
            logger.info(">>> RETRIEVAL: single-scope fallback omit domain filter (scoped doc)")
    if not merged and scoped_doc and technology:
        wide_top = max(
            (initial_top_k if enable_reranker else max_chunks_no_rerank) * 2,
            rerank_top_n * 4,
        )
        vector_results = vector_store.search(
            query_embedding=query_embedding,
            top_k=wide_top,
            technology=None,
            domain=None,
            document_id=scoped_doc,
            file_id=scoped_doc,
            query_text=search_query,
        )
        keyword_results = []
        if enable_keyword and keyword_service:
            keyword_results = keyword_service.search(
                query=keyword_query,
                top_k=max(keyword_top_k, wide_top),
                technology=None,
                domain=None,
                document_id=scoped_doc,
                file_id=scoped_doc,
            )
        merged = _merge_hybrid_results(vector_results, keyword_results, vector_weight, keyword_weight)
        merged = _apply_retrieval_relevance_tuning(merged, topic_terms or [])
        if merged:
            logger.info(">>> RETRIEVAL: single-scope document-wide search (technology labels mismatch ingest)")
    logger.info(
        ">>> RETRIEVAL: vector=%s keyword=%s merged=%s",
        len(vector_results), len(keyword_results), len(merged)
    )

    if enable_reranker and len(merged) > rerank_top_n:
        logger.info(">>> RERANK: input_chunks=%s -> keeping top_n=%s", len(merged), rerank_top_n)
        reranker = get_reranker_service()
        context_chunks = reranker.rerank(query=rerank_query, chunks=merged, top_n=rerank_top_n)
        logger.info(">>> RERANK: output_chunks=%s (sent to LLM)", len(context_chunks))
    else:
        context_chunks = merged[:rerank_top_n] if enable_reranker else merged[:max_chunks_no_rerank]
        logger.info(
            ">>> RETRIEVAL: rerank skipped (chunks=%s), using %s chunks for LLM",
            len(merged), len(context_chunks),
        )
    return context_chunks


def _normalize_similarity(score: float) -> float:
    """
    Normalize a vector search score into an approximate similarity in [0, 1].
    If score is already in [0,1], return it.
    If score > 1, treat as distance-like and map to (0,1] via 1/(1+score).
    """
    try:
        s = float(score)
    except Exception:
        return 0.0
    if s < 0:
        return 0.0
    if s <= 1.0:
        return s
    return 1.0 / (1.0 + s)


def _compute_retrieval_confidence(context_chunks: list) -> tuple:
    """
    Compute average similarity and spread from context chunks.
    Returns (avg_similarity, spread, count).
    """
    if not context_chunks:
        return 0.0, 0.0, 0
    sims = []
    for c in context_chunks:
        score = c.get("score", 0.0)
        sims.append(_normalize_similarity(score))
    if not sims:
        return 0.0, 0.0, 0
    avg_sim = sum(sims) / len(sims)
    spread = max(sims) - min(sims) if len(sims) > 1 else 0.0
    return avg_sim, spread, len(sims)


def _insufficient_context_message(available_technologies: Optional[list] = None) -> str:
    base = "I couldn't find enough information in the uploaded documents to answer that."
    if available_technologies:
        topics = ", ".join(available_technologies)
        return f"{base} Available topics: {topics}."
    return base


def _normalize_bm25(score: float) -> float:
    """
    Normalize BM25 score (lower is better) into similarity in (0,1].
    """
    try:
        s = float(score)
    except Exception:
        return 0.0
    if s < 0:
        return 0.0
    return 1.0 / (1.0 + s)


def _chunk_key(item: dict) -> str:
    chunk_id = item.get("chunk_id")
    if chunk_id:
        return f"chunk:{chunk_id}"
    doc_id = item.get("document_id", "")
    chunk_index = item.get("chunk_index", "")
    file_name = item.get("file_name", "")
    return f"{doc_id}:{chunk_index}:{file_name}"


def _merge_hybrid_results(
    vector_results: list,
    keyword_results: list,
    vector_weight: float,
    keyword_weight: float,
) -> list:
    merged = {}

    # Add vector results
    for v in vector_results:
        key = _chunk_key(v)
        vec_sim = _normalize_similarity(v.get("score", 0.0))
        item = merged.get(key, dict(v))
        item["vector_score"] = vec_sim
        item["keyword_score"] = item.get("keyword_score", 0.0)
        item["hybrid_score"] = item.get("hybrid_score", 0.0) + (vector_weight * vec_sim)
        # Use hybrid score as primary score for downstream gating/sorting
        item["score"] = item["hybrid_score"]
        merged[key] = item

    # Add keyword results
    for k in keyword_results:
        key = _chunk_key(k)
        kw_sim = _normalize_bm25(k.get("score", 0.0))
        item = merged.get(key, dict(k))
        item["vector_score"] = item.get("vector_score", 0.0)
        item["keyword_score"] = kw_sim
        item["hybrid_score"] = item.get("hybrid_score", 0.0) + (keyword_weight * kw_sim)
        item["score"] = item["hybrid_score"]
        merged[key] = item

    merged_list = list(merged.values())
    merged_list.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    return merged_list


def _query_seeks_figure_or_image(query: str) -> bool:
    """True when user likely wants to see a diagram/figure file, not only text."""
    q = (query or "").lower()
    if not q.strip():
        return False
    visual = (
        "image", "figure", "diagram", "picture", "chart", "graph", "plot",
        "screenshot", "illustration", "show me", "show the", "see the", "view the",
        "display the", "can i see",
    )
    if not any(v in q for v in visual):
        return False
    return True


def _maybe_supplement_image_caption_chunks(
    context_chunks: list,
    query: str,
    scope_file_ids: list,
    technologies: list,
) -> list:
    """
    Vector search often ranks text above image_caption chunks. When the user asks to *see* a figure,
    pull image_caption rows from the chunk store for scoped documents so RAW IMAGE + view_url reach the LLM.
    """
    if not _query_seeks_figure_or_image(query):
        return context_chunks

    chunk_store = get_chunk_store()
    seen = {c.get("chunk_id") for c in context_chunks if c.get("chunk_id")}
    doc_candidates: list = []
    if scope_file_ids:
        doc_candidates = list(scope_file_ids)
    else:
        doc_candidates = list({c.get("document_id") for c in context_chunks if c.get("document_id")})
        doc_candidates = doc_candidates[:10]
    if len(doc_candidates) > 8:
        doc_candidates = doc_candidates[:8]

    tech_fallback = technologies[0] if len(technologies) == 1 else None
    added: list = []
    per_doc = 15
    for did in doc_candidates:
        if not did:
            continue
        try:
            rows = chunk_store.list_chunks_by_type(did, "image_caption", limit=per_doc)
        except Exception as e:
            logger.warning(">>> RETRIEVAL: list image_caption chunks failed doc=%s: %s", did[:8], e)
            continue
        for row in rows:
            cid = row.get("chunk_id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            mjson = row.get("metadata_json") or "{}"
            added.append({
                "text": row.get("retrieval_text") or "",
                "chunk_id": cid,
                "document_id": did,
                "chunk_type": "image_caption",
                "metadata_json": mjson,
                "page_number": row.get("page_number", -1),
                "section_title": row.get("section_title", ""),
                "technology": tech_fallback or "general",
                "domain": None,
                "score": 0.55,
                "hybrid_score": 0.55,
            })
    if not added:
        return context_chunks
    logger.info(">>> RETRIEVAL: supplemented %s image_caption chunks for figure/image query", len(added))
    return list(context_chunks) + added


def _resolve_parent_context(context_chunks: list) -> list:
    """Hierarchical retrieval: replace child snippets with section parent context."""
    if not context_chunks:
        return context_chunks
    if not getattr(settings, "ENABLE_PARENT_CHILD_RETRIEVAL", True):
        return context_chunks
    from app.services.parent_context_resolution import resolve_parent_context

    return resolve_parent_context(context_chunks)


def _expand_context_graph(
    context_chunks: list,
    query: str = "",
    scope_file_ids: Optional[list] = None,
    technologies: Optional[list] = None,
) -> list:
    """Phase 4: prev/next, cross-type links, document_edges, token-budget pack."""
    if not context_chunks:
        return context_chunks
    if not getattr(settings, "ENABLE_GRAPH_CONTEXT_EXPANSION", True):
        return context_chunks
    from app.services.graph_context_expansion import expand_context_graph

    seeks_figure = _query_seeks_figure_or_image(query)
    expanded = expand_context_graph(
        context_chunks,
        query=query,
        max_hops=1,
        query_seeks_figure=seeks_figure,
    )
    if seeks_figure:
        has_figure = any(
            (c.get("chunk_type") or "").lower() == "image_caption" for c in expanded
        )
        if not has_figure:
            expanded = _maybe_supplement_image_caption_chunks(
                expanded,
                query,
                scope_file_ids or [],
                technologies or [],
            )
    return expanded


def _hydrate_chunks(context_chunks: list) -> list:
    """
    Enrich chunks with metadata, chunk_type, and raw table/code/image content.
    Prefer chunk_store for metadata when chunk_id is present so raw_code_id/raw_table_id
    are never lost to vector-store truncation.
    """
    if not context_chunks:
        return context_chunks

    chunk_store = get_chunk_store()
    raw_store = get_raw_block_store()

    chunk_ids = [c.get("chunk_id") for c in context_chunks if c.get("chunk_id")]
    chunk_map = chunk_store.get_chunks(chunk_ids) if chunk_ids else {}

    for c in context_chunks:
        cid = c.get("chunk_id")
        if cid and cid in chunk_map:
            row = chunk_map[cid]
            c["metadata_json"] = row.get("metadata_json", "")
            c["chunk_type"] = row.get("chunk_type", "paragraph")
            c["page_number"] = row.get("page_number", -1)
            c["section_title"] = row.get("section_title", "")
            c["text"] = row.get("retrieval_text", "") or c.get("text", "")

        meta = {}
        if c.get("metadata_json"):
            try:
                meta = json.loads(c.get("metadata_json"))
            except Exception:
                meta = {}
        c["metadata"] = meta

        chunk_type = c.get("chunk_type")
        if chunk_type == "table_summary":
            raw_id = meta.get("raw_table_id")
            if raw_id:
                c["raw_table"] = raw_store.get_table(raw_id)
        elif chunk_type == "code_summary":
            raw_id = meta.get("raw_code_id")
            if raw_id:
                c["raw_code"] = raw_store.get_code(raw_id)
        elif chunk_type == "image_caption":
            raw_id = meta.get("raw_image_id")
            img = None
            if raw_id:
                img = raw_store.get_image(raw_id)
            from app.services.figure_serving import (
                build_figure_dict,
                hydrate_raw_image_fallback,
            )

            if img:
                img = dict(img)
                did = (img.get("document_id") or c.get("document_id") or "").strip()
                fig = build_figure_dict(
                    document_id=did,
                    page_number=img.get("page_number") or c.get("page_number"),
                    image_id=img.get("image_id"),
                    caption=img.get("caption") or c.get("text") or "",
                    existing_view_url=img.get("view_url"),
                    image_path=img.get("image_path"),
                )
                if fig:
                    img.update(fig)
                c["raw_image"] = img
            else:
                fallback = hydrate_raw_image_fallback(
                    c, document_id=c.get("document_id")
                )
                if fallback:
                    c["raw_image"] = fallback

    return context_chunks


def _sources_from_chunks(context_chunks: list) -> list:
    sources = []
    for c in context_chunks:
        tech = c.get("technology") or c.get("file_name")
        if tech and tech not in sources:
            sources.append(tech if isinstance(tech, str) else str(tech))
    return list(sources)[:10]


def _figure_relevance_score(chunk: Dict[str, Any], query: str) -> float:
    from app.services.figure_serving import _query_figure_relevance_score

    base = float(chunk.get("score") or chunk.get("hybrid_score") or 0.0)
    text = (chunk.get("text") or "") + " " + (chunk.get("raw_image") or {}).get(
        "caption", ""
    )
    if (chunk.get("chunk_type") or "").lower() != "image_caption":
        return base - 2.0 + _query_figure_relevance_score(text, query) * 0.5
    return base + _query_figure_relevance_score(text, query)


def _figure_row_from_raw_image(
    raw: Dict[str, Any],
    *,
    fallback_text: str = "",
    page_number: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Build a figure dict with a view_url that serves from disk when possible."""
    from app.services.figure_serving import build_figure_dict

    did = (raw.get("document_id") or "").strip()
    if not did:
        return None
    return build_figure_dict(
        document_id=did,
        page_number=page_number if page_number is not None else raw.get("page_number"),
        image_id=raw.get("image_id"),
        caption=(raw.get("caption") or fallback_text or "Figure from document"),
        existing_view_url=raw.get("view_url"),
        image_path=raw.get("image_path"),
    )


def _figures_from_context_chunks(context_chunks: list, query: str = "") -> list:
    """Build figure payloads from hydrated retrieval hits (best match only)."""
    seen_pages: set = set()
    seen_urls: set = set()
    figures: list = []
    candidates = [
        c
        for c in context_chunks
        if (c.get("raw_image") or {}).get("view_url")
        or (c.get("raw_image") or {}).get("image_id")
    ]
    candidates.sort(key=lambda c: _figure_relevance_score(c, query), reverse=True)
    for c in candidates:
        raw = c.get("raw_image") or {}
        fig = _figure_row_from_raw_image(
            raw,
            fallback_text=(c.get("text") or ""),
            page_number=c.get("page_number"),
        )
        if not fig or not fig.get("view_url"):
            continue
        page_key = (fig.get("document_id"), fig.get("page_number"))
        if page_key in seen_pages:
            continue
        view_url = fig["view_url"]
        if view_url in seen_urls:
            continue
        seen_pages.add(page_key)
        seen_urls.add(view_url)
        figures.append(fig)
        if len(figures) >= 1:
            break
    return figures


def _figures_from_scoped_documents(scope_file_ids: list, query: str = "") -> list:
    """
    Load all image_caption chunks for scoped documents (fallback when retrieval
    hits do not carry raw_image on the in-memory chunk dict).
    """
    if not scope_file_ids:
        return []
    chunk_store = get_chunk_store()
    raw_store = get_raw_block_store()
    seen: set = set()
    figures: list = []
    q = (query or "").lower()

    for did in _cap_scope_document_ids(scope_file_ids)[:5]:
        try:
            rows = chunk_store.list_chunks_by_type(did, "image_caption", limit=25)
        except Exception as e:
            logger.warning(">>> FIGURES: list image_caption failed doc=%s: %s", did[:8], e)
            continue

        def _row_score(row: Dict[str, Any]) -> float:
            text = (row.get("retrieval_text") or "").lower()
            score = 0.1
            for term in re.findall(r"[a-z]{4,}", q):
                if term in text:
                    score += 0.4
            if "visual description" in text:
                score += 0.3
            if "lifecycle" in q and "lifecycle" in text:
                score += 0.5
            if "diagram" in q and "diagram" in text:
                score += 0.5
            try:
                score += max(0, 0.05 * int(row.get("page_number") or 0))
            except (TypeError, ValueError):
                pass
            return score

        rows.sort(key=_row_score, reverse=True)
        for row in rows:
            meta: Dict[str, Any] = {}
            try:
                meta = json.loads(row.get("metadata_json") or "{}")
            except Exception:
                meta = {}
            raw_id = meta.get("raw_image_id")
            if not raw_id:
                continue
            img = raw_store.get_image(raw_id)
            fig = None
            if img:
                fig = _figure_row_from_raw_image(
                    img,
                    fallback_text=row.get("retrieval_text") or "",
                    page_number=row.get("page_number"),
                )
            if not fig:
                from app.services.figure_serving import figure_from_page

                page = row.get("page_number")
                doc_id = row.get("document_id") or did
                if page is not None:
                    fig = figure_from_page(
                        doc_id,
                        int(page),
                        caption=row.get("retrieval_text") or "",
                        image_id=raw_id,
                    )
            page_key = (fig.get("document_id"), fig.get("page_number"))
            if not fig or fig["view_url"] in seen or page_key in seen:
                continue
            seen.add(fig["view_url"])
            seen.add(page_key)
            figures.append(fig)
            if len(figures) >= 1:
                return figures
    if not figures:
        from app.services.figure_serving import figures_from_disk

        figures = figures_from_disk(
            _cap_scope_document_ids(scope_file_ids),
            query=query,
            max_figures=1,
        )
    return figures


def _resolve_figures_for_query(
    context_chunks: list,
    scope_file_ids: list,
    query: str,
) -> list:
    """Prefer hydrated retrieval hits; fall back to all image_caption rows in scope."""
    figures = _figures_from_context_chunks(context_chunks, query=query)
    if not figures and _query_seeks_figure_or_image(query):
        figures = _figures_from_scoped_documents(scope_file_ids, query=query)
    if not figures and _query_seeks_figure_or_image(query) and scope_file_ids:
        from app.services.figure_serving import figures_from_disk

        figures = figures_from_disk(
            _cap_scope_document_ids(scope_file_ids), query=query, max_figures=1
        )
    if _query_seeks_figure_or_image(query):
        logger.info(
            ">>> FIGURES: resolved %s for query (context_hits_with_raw_image=%s)",
            len(figures),
            sum(1 for c in context_chunks if (c.get("raw_image") or {}).get("image_id")),
        )
    return figures


def _answer_has_embedded_document_image(answer: str) -> bool:
    """True only when answer already embeds our API image route (not external URLs)."""
    return bool(
        re.search(
            r"!\[[^\]]*\]\(/documents/[^)/\s]+/images/(?:page/\d+|[^)/\s]+)\)",
            answer or "",
        )
    )


def _ensure_document_figures_in_answer(answer: str, figures: list, query: str) -> str:
    """
    When the user asked to see a diagram, append Markdown image lines for stored figures
    if the LLM omitted them (e.g. linked an external URL from PDF text instead).
    """
    if not figures or not _query_seeks_figure_or_image(query):
        return answer
    if _answer_has_embedded_document_image(answer):
        return answer
    lines = ["\n\n## Diagram from your document\n"]
    for fig in figures[:1]:
        alt = "Figure"
        cap = (fig.get("caption") or "").strip()
        if cap:
            alt = cap[:120] + ("…" if len(cap) > 120 else "")
        lines.append(f"![{alt}]({fig['view_url']})")
    return (answer or "").rstrip() + "\n".join(lines)


def _sse_payload(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _friendly_stream_llm_error(exc: Exception) -> str:
    """Short user-facing message for SSE error events."""
    msg = str(exc) or type(exc).__name__
    low = msg.lower()
    if "rate_limit" in low or "429" in msg:
        if "tokens per day" in low or "tpd" in low:
            return (
                "Groq daily token limit reached for this API key. "
                "Wait about 40 minutes, switch LLM_PROVIDER=openai in .env, or upgrade Groq billing."
            )
        return (
            "Groq rate limit hit (too many requests). Wait a minute and try again, "
            "or reduce RERANK / context size in .env."
        )
    if "413" in msg or "too large" in low or "request too large" in low:
        return "Prompt too large for Groq. Lower GROQ_MAX_CONTEXT_CHARS or scope fewer documents."
    return msg[:800]


async def _rag_turn_async(request: ChatRequest) -> dict:
    """
    Scope resolution, retrieval, hydration. Returns metadata for LLM (sync or stream).
    """
    query = (request.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    logger.info(">>> CHAT: query=%s", repr(query[:80] + ("..." if len(query) > 80 else "")))
    chat_service = get_chat_service()

    # Resolve or create session
    session_id = request.session_id
    if not session_id:
        session = chat_service.create_session(title="New Chat")
        session_id = session["id"]
        logger.info(">>> CHAT: created new session_id=%s", session_id[:8] + "...")
    else:
        logger.info(">>> CHAT: using session_id=%s", session_id[:8] + "...")

    # Ensure session exists if session_id was provided
    if request.session_id and not chat_service.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    # ==================== DOCUMENT SCOPE (SEGREGATION) ====================
    # Prefer explicit scope from request. If absent, fall back to session-attached docs.
    requested_scope = []
    if getattr(request, "file_ids", None):
        requested_scope = [fid for fid in (request.file_ids or []) if fid]
    elif request.file_id:
        requested_scope = [request.file_id]

    # Persist scope on the server whenever the client sends it (so message 2+ works
    # even if the UI stops re-sending file_ids or session_documents was never PUT).
    if requested_scope:
        try:
            chat_service.set_session_documents(session_id, requested_scope)
        except Exception as e:
            logger.warning(">>> CHAT: could not persist session documents: %s", e)

    session_scope = []
    if not requested_scope:
        try:
            session_scope = chat_service.get_session_documents(session_id)
        except Exception:
            session_scope = []

    scope_file_ids = requested_scope or session_scope

    if not scope_file_ids and settings.CHAT_AUTO_SCOPE_ALL_INGESTED:
        try:
            auto_ids = get_document_store().list_ingested_document_ids()
            if auto_ids:
                scope_file_ids = auto_ids
                logger.info(
                    ">>> CHAT: auto scope — all ingested documents (count=%s)",
                    len(scope_file_ids),
                )
        except Exception as e:
            logger.warning(">>> CHAT: auto scope lookup failed: %s", e)

    if not scope_file_ids:
        if settings.CHAT_AUTO_SCOPE_ALL_INGESTED:
            detail = (
                "No searchable documents yet. Upload PDFs from the Library or Upload tab "
                "and wait until ingestion completes."
            )
        else:
            detail = (
                "No document scope provided. Send file_ids, or attach documents via "
                "/sessions/{session_id}/documents."
            )
        raise HTTPException(status_code=400, detail=detail)

    # Only record the user message after scope is valid (avoids orphan messages on 400).
    chat_service.add_message(session_id, "user", query)

    # Classify query: prefer scoped document ingest metadata over LLM multi-tech guesses
    recent_messages = chat_service.get_context_messages(session_id, context_window=6)
    scoped_techs, scoped_domain = _technologies_from_scoped_documents(scope_file_ids)
    from_scoped_documents = bool(scoped_techs)
    if scoped_techs:
        technologies, domain = scoped_techs, scoped_domain
        from_conversation_context = False
        # Retrieval uses document_id pool only; avoid multi-tech balanced loops per doc.
        if len(technologies) > 1:
            logger.info(
                ">>> RETRIEVAL: scoped docs have mixed ingest tags %s — pool search ignores tech filter",
                technologies,
            )
        logger.info(
            ">>> RETRIEVAL: query_classification from_scoped_documents technologies=%s domain=%s docs=%s",
            technologies, domain, len(scope_file_ids),
        )
    else:
        technologies, domain, from_conversation_context = classify_query_with_context(query, recent_messages)
        logger.info(
            ">>> RETRIEVAL: query_classification technologies=%s domain=%s from_conversation_context=%s",
            technologies, domain, from_conversation_context,
        )

    # If context-based classification yields multiple technologies, try to narrow to the last user topic
    topic_context_text = ""
    if from_conversation_context and (not technologies or len(technologies) > 1):
        prev_text, prev_techs, prev_domain = _get_last_user_message_with_tech(recent_messages)
        if prev_text and prev_techs:
            topic_context_text = prev_text
            technologies = prev_techs
            domain = prev_domain
            logger.info(
                ">>> RETRIEVAL: narrowed_from_last_user_with_tech technologies=%s domain=%s (prev_user=%s)",
                technologies, domain, repr(prev_text[:80] + ("..." if len(prev_text) > 80 else "")),
            )

    topic_terms = _topic_terms(query, technologies)
    if _is_short_follow_up(query):
        prev_user = _get_previous_user_message(recent_messages)
        if prev_user:
            topic_terms = list(
                dict.fromkeys(topic_terms + _topic_terms(prev_user, technologies))
            )

    search_query = _build_search_query(
        query,
        technologies,
        from_conversation_context,
        recent_messages,
        topic_context_text,
        from_scoped_documents=from_scoped_documents,
    )
    keyword_query = _build_keyword_query(
        query, recent_messages, technologies, from_scoped_documents=from_scoped_documents
    )
    rerank_query = _build_rerank_query(query, recent_messages)
    if search_query.strip() != query.strip():
        logger.info(
            ">>> RETRIEVAL: embedding_search_query=%s",
            repr(search_query[:100] + ("..." if len(search_query) > 100 else "")),
        )
    if topic_terms:
        logger.info(">>> RETRIEVAL: topic_terms=%s", topic_terms)

    keyword_service = get_keyword_search_service()
    enable_reranker = getattr(settings, "ENABLE_RERANKER", True)
    initial_top_k = getattr(settings, "RERANK_INITIAL_TOP_K", 20)
    rerank_top_n = getattr(settings, "RERANK_TOP_N", 5)
    max_chunks_no_rerank = getattr(settings, "MAX_CHUNKS_TO_RETRIEVE", 5)
    enable_keyword = getattr(settings, "ENABLE_KEYWORD_SEARCH", True)
    keyword_top_k = getattr(settings, "KEYWORD_TOP_K", initial_top_k)
    vector_weight = getattr(settings, "HYBRID_VECTOR_WEIGHT", 0.7)
    keyword_weight = getattr(settings, "HYBRID_KEYWORD_WEIGHT", 0.3)
    # Vector store is optional; fall back to keyword-only if Zilliz is stopped/unavailable.
    vector_store = None
    embedding_service = None
    try:
        vector_store = get_vector_store()
        embedding_service = get_embedding_service()
    except HTTPException as e:
        # If vector DB is down, keep the app usable via keyword search.
        logger.warning(">>> RETRIEVAL: vector store unavailable (%s). Falling back to keyword-only.", e.detail)
        vector_store = None
        embedding_service = None

    if vector_store is None or embedding_service is None:
        if not enable_keyword or keyword_service is None:
            raise HTTPException(status_code=503, detail="Retrieval is unavailable (vector DB down and keyword search disabled).")
        context_chunks = []
        doc_ids = _cap_scope_document_ids(scope_file_ids)
        if keyword_service and doc_ids:
            hits = await asyncio.to_thread(
                keyword_service.search,
                query=keyword_query,
                top_k=keyword_top_k,
                technology=None,
                domain=None,
                document_ids=doc_ids,
            )
            context_chunks = _apply_retrieval_relevance_tuning(hits or [], topic_terms)
            context_chunks.sort(key=lambda x: x.get("score", 0.0))
            context_chunks = context_chunks[:rerank_top_n]
        logger.info(
            ">>> RETRIEVAL: keyword-only scoped pool docs=%s context_chunks=%s",
            len(doc_ids),
            len(context_chunks),
        )
    else:
        context_chunks = await _retrieve_scoped_document_pool_async(
            search_query=search_query,
            keyword_query=keyword_query,
            rerank_query=rerank_query,
            scope_file_ids=scope_file_ids,
            vector_store=vector_store,
            embedding_service=embedding_service,
            keyword_service=keyword_service,
            enable_keyword=enable_keyword,
            keyword_top_k=keyword_top_k,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            enable_reranker=enable_reranker,
            initial_top_k=initial_top_k,
            rerank_top_n=rerank_top_n,
            topic_terms=topic_terms,
        )
        logger.info(
            ">>> RETRIEVAL: scoped docs=%s final context_chunks=%s",
            len(scope_file_ids),
            len(context_chunks),
        )

    # Self-refining retrieval loop (single retry with query rewrite)
    enable_query_rewrite = getattr(settings, "ENABLE_QUERY_REWRITE", True)
    if enable_query_rewrite:
        llm_service = get_llm_service()
        try:
            sufficient = llm_service.is_context_sufficient(query, context_chunks)
        except Exception as e:
            logger.warning(">>> RETRIEVAL: sufficiency check failed, skipping rewrite: %s", e)
            sufficient = True

        if not sufficient:
            try:
                rewritten = llm_service.rewrite_query(query, chat_history=recent_messages)
            except Exception as e:
                logger.warning(">>> RETRIEVAL: rewrite failed, skipping retry: %s", e)
                rewritten = query

            if rewritten and rewritten.strip() and rewritten.strip() != query.strip():
                rewrite_search_query = _build_search_query(
                    rewritten,
                    technologies,
                    from_conversation_context,
                    recent_messages,
                    topic_context_text,
                    from_scoped_documents=from_scoped_documents,
                )
                rewrite_keyword_query = _build_keyword_query(
                    rewritten,
                    recent_messages,
                    technologies,
                    from_scoped_documents=from_scoped_documents,
                )
                rewrite_topic_terms = _topic_terms(rewritten, technologies)
                logger.info(
                    ">>> RETRIEVAL: rewrite_query=%s",
                    repr(rewritten[:120] + ("..." if len(rewritten) > 120 else ""))
                )
                if vector_store is None or embedding_service is None:
                    doc_ids_rw = _cap_scope_document_ids(scope_file_ids)
                    ctx_rw = (
                        await asyncio.to_thread(
                            keyword_service.search,
                            query=rewrite_keyword_query,
                            top_k=keyword_top_k,
                            technology=None,
                            domain=None,
                            document_ids=doc_ids_rw,
                        )
                        if keyword_service
                        else []
                    )
                    ctx_rw = _apply_retrieval_relevance_tuning(ctx_rw or [], rewrite_topic_terms)
                    ctx_rw.sort(key=lambda x: x.get("score", 0.0))
                    context_chunks = ctx_rw[:rerank_top_n]
                    logger.info(
                        ">>> RETRIEVAL: keyword-only rewrite pool docs=%s context_chunks=%s",
                        len(doc_ids_rw),
                        len(context_chunks),
                    )
                else:
                    context_chunks = await _retrieve_scoped_document_pool_async(
                        search_query=rewrite_search_query,
                        keyword_query=rewrite_keyword_query,
                        rerank_query=rewritten,
                        scope_file_ids=scope_file_ids,
                        vector_store=vector_store,
                        embedding_service=embedding_service,
                        keyword_service=keyword_service,
                        enable_keyword=enable_keyword,
                        keyword_top_k=keyword_top_k,
                        vector_weight=vector_weight,
                        keyword_weight=keyword_weight,
                        enable_reranker=enable_reranker,
                        initial_top_k=initial_top_k,
                        rerank_top_n=rerank_top_n,
                        topic_terms=rewrite_topic_terms,
                    )
                    logger.info(
                        ">>> RETRIEVAL: rewrite scoped pool docs=%s context_chunks=%s",
                        len(scope_file_ids),
                        len(context_chunks),
                    )

    # Phase 4 graph expansion (+ figure fallback via illustrates edges or supplement)
    context_chunks = await asyncio.to_thread(
        _expand_context_graph,
        context_chunks,
        query=query,
        scope_file_ids=scope_file_ids,
        technologies=technologies,
    )

    # Hydrate chunks with raw data (tables/code/images)
    context_chunks = await asyncio.to_thread(_hydrate_chunks, context_chunks)

    # Parent–child: child hit → section parent text for LLM (Small-to-Big)
    context_chunks = await asyncio.to_thread(_resolve_parent_context, context_chunks)

    # Chat history for context window
    chat_history = chat_service.get_context_messages(session_id, context_window=6)
    available_technologies = get_document_store().get_technologies()
    logger.info(">>> LLM: context_chunks=%s, chat_history_messages=%s, available_technologies=%s", len(context_chunks), len(chat_history), available_technologies)

    # Retrieval confidence gate
    avg_similarity, score_spread, chunk_count = _compute_retrieval_confidence(context_chunks)
    logger.info(
        ">>> RETRIEVAL: confidence avg_similarity=%.4f spread=%.4f chunks=%s",
        avg_similarity, score_spread, chunk_count,
    )
    # Marginal similarity on huge PDFs often clusters ~0.22–0.28; avoid blocking near threshold.
    if avg_similarity < 0.22 or chunk_count < 2:
        return {
            "session_id": session_id,
            "query": query,
            "technologies": technologies,
            "domain": domain,
            "scope_file_ids": scope_file_ids,
            "context_chunks": context_chunks,
            "chat_service": chat_service,
            "chat_history": chat_history,
            "available_technologies": available_technologies,
            "insufficient_answer": _insufficient_context_message(available_technologies),
        }

    return {
        "session_id": session_id,
        "query": query,
        "technologies": technologies,
        "domain": domain,
        "scope_file_ids": scope_file_ids,
        "context_chunks": context_chunks,
        "chat_service": chat_service,
        "chat_history": chat_history,
        "available_technologies": available_technologies,
        "insufficient_answer": None,
    }


def _record_llm_usage(
    *,
    session_id: str,
    message_id,
    query: str,
    llm_service: LLMService,
) -> None:
    usage = getattr(llm_service, "last_usage", None) or {}
    prompt_toks = usage.get("prompt_tokens")
    completion_toks = usage.get("completion_tokens")
    total_toks = usage.get("total_tokens")
    provider = (settings.LLM_PROVIDER or "").lower().strip()
    model_used = getattr(llm_service, "model", None)

    cost_usd = None
    try:
        if prompt_toks is not None and completion_toks is not None:
            if provider == "openai":
                in_rate = getattr(settings, "OPENAI_INPUT_USD_PER_1M", None)
                out_rate = getattr(settings, "OPENAI_OUTPUT_USD_PER_1M", None)
                if in_rate is not None and out_rate is not None:
                    cost_usd = (float(prompt_toks) / 1_000_000.0) * float(in_rate) + (
                        float(completion_toks) / 1_000_000.0
                    ) * float(out_rate)
            elif provider == "groq":
                in_rate = float(getattr(settings, "GROQ_INPUT_USD_PER_1M", 0.59))
                out_rate = float(getattr(settings, "GROQ_OUTPUT_USD_PER_1M", 0.79))
                cost_usd = (float(prompt_toks) / 1_000_000.0) * in_rate + (
                    float(completion_toks) / 1_000_000.0
                ) * out_rate
    except Exception:
        cost_usd = None

    logger.info(
        ">>> TOKENS: session=%s msg_id=%s provider=%s model=%s prompt=%s completion=%s total=%s cost_usd=%s",
        session_id,
        message_id,
        provider,
        model_used,
        prompt_toks,
        completion_toks,
        total_toks,
        f"{cost_usd:.6f}" if isinstance(cost_usd, (int, float)) else None,
    )

    record_chat_completion(
        session_id=session_id,
        message_id=message_id,
        query_preview=query,
        prompt_tokens=prompt_toks,
        completion_tokens=completion_toks,
        total_tokens=total_toks,
        model=model_used,
        provider=provider or "openai",
        cost_usd=float(cost_usd) if isinstance(cost_usd, (int, float)) else None,
    )


def _chat_response_from_turn(turn: dict, answer: str, query: str = "") -> ChatResponse:
    technologies = turn.get("technologies") or []
    domain = turn.get("domain")
    context_chunks = turn.get("context_chunks") or []
    scope_file_ids = turn.get("scope_file_ids") or []
    sources = _sources_from_chunks(context_chunks)
    figures = _resolve_figures_for_query(context_chunks, scope_file_ids, query)
    answer = _ensure_document_figures_in_answer(answer, figures, query)
    detected_technology_str = ", ".join(technologies) if technologies else None
    return ChatResponse(
        answer=answer,
        session_id=turn["session_id"],
        sources=sources if sources else None,
        detected_technology=detected_technology_str,
        detected_domain=domain,
        figures=figures if figures else None,
    )


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """RAG chat (non-streaming). Creates a session when session_id is omitted."""
    turn = await _rag_turn_async(request)
    chat_service = turn["chat_service"]
    session_id = turn["session_id"]
    query = turn["query"]

    if turn.get("insufficient_answer"):
        answer = turn["insufficient_answer"]
        chat_service.add_message(session_id, "assistant", answer)
        return _chat_response_from_turn(turn, answer, query=query)

    llm_service = get_llm_service()
    answer = llm_service.generate_response(
        query=query,
        context_chunks=turn["context_chunks"],
        chat_history=turn["chat_history"],
        available_technologies=turn["available_technologies"],
    )
    assistant_row = chat_service.add_message(session_id, "assistant", answer)
    _record_llm_usage(
        session_id=session_id,
        message_id=assistant_row.get("id"),
        query=query,
        llm_service=llm_service,
    )
    logger.info(">>> CHAT: done session_id=%s", session_id[:8] + "...")
    return _chat_response_from_turn(turn, answer, query=query)


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """RAG chat with SSE token streaming for the assistant reply."""
    turn = await _rag_turn_async(request)
    chat_service = turn["chat_service"]
    session_id = turn["session_id"]
    query = turn["query"]
    technologies = turn.get("technologies") or []
    domain = turn.get("domain")
    detected_technology_str = ", ".join(technologies) if technologies else None
    scope_file_ids = turn.get("scope_file_ids") or []
    context_chunks_for_figures = turn.get("context_chunks") or []
    stream_figures = _resolve_figures_for_query(
        context_chunks_for_figures, scope_file_ids, query
    )

    async def event_generator():
        meta = {
            "type": "meta",
            "session_id": session_id,
            "detected_technology": detected_technology_str,
            "detected_domain": domain,
            "figures": stream_figures if stream_figures else None,
        }
        yield _sse_payload(meta)

        if turn.get("insufficient_answer"):
            answer = turn["insufficient_answer"]
            chat_service.add_message(session_id, "assistant", answer)
            yield _sse_payload({"type": "token", "content": answer})
            yield _sse_payload(
                {
                    "type": "done",
                    "answer": answer,
                    "session_id": session_id,
                    "sources": None,
                    "detected_technology": detected_technology_str,
                    "detected_domain": domain,
                }
            )
            return

        llm_service = get_llm_service()
        parts: list = []
        try:
            for delta in llm_service.stream_response(
                query=query,
                context_chunks=turn["context_chunks"],
                chat_history=turn["chat_history"],
                available_technologies=turn["available_technologies"],
            ):
                parts.append(delta)
                yield _sse_payload({"type": "token", "content": delta})
        except Exception as e:
            logger.exception(">>> CHAT STREAM: LLM error: %s", e)
            yield _sse_payload(
                {"type": "error", "detail": _friendly_stream_llm_error(e)}
            )
            return

        answer = "".join(parts).strip()
        figures = stream_figures or _resolve_figures_for_query(
            turn.get("context_chunks") or [],
            scope_file_ids,
            query,
        )
        answer = _ensure_document_figures_in_answer(answer, figures, query)
        assistant_row = chat_service.add_message(session_id, "assistant", answer)
        _record_llm_usage(
            session_id=session_id,
            message_id=assistant_row.get("id"),
            query=query,
            llm_service=llm_service,
        )
        sources = _sources_from_chunks(turn.get("context_chunks") or [])
        if figures:
            logger.info(">>> CHAT STREAM: attaching %s document figure(s) to reply", len(figures))
        logger.info(">>> CHAT STREAM: done session_id=%s", session_id[:8] + "...")
        yield _sse_payload(
            {
                "type": "done",
                "answer": answer,
                "session_id": session_id,
                "sources": sources if sources else None,
                "detected_technology": detected_technology_str,
                "detected_domain": domain,
                "figures": figures if figures else None,
            }
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
