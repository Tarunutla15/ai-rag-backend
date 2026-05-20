"""OpenAI embedding service with batch processing support."""
import asyncio
import time
from typing import List

from openai import AsyncOpenAI, OpenAI


class EmbeddingService:
    """Service for generating embeddings using OpenAI with batch processing."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize embedding service.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model name
        """
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.model = model
        # Default batch size: 100 chunks per batch
        # Each chunk ~1000 chars ≈ ~250 tokens, so 100 chunks ≈ 25k tokens (well under 300k limit)
        self.default_batch_size = 100
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    def generate_embeddings(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Processes texts in batches to avoid OpenAI's token limit (300k tokens per request).
        For large PDFs with many chunks, this prevents "max_tokens_per_request" errors.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process per batch (default: 100)
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        if batch_size is None:
            batch_size = self.default_batch_size
        
        # If small number of texts, process all at once
        if len(texts) <= batch_size:
            print(f">>> EMBEDDING: Processing {len(texts)} chunks in single batch", flush=True)
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                print(f">>> EMBEDDING ERROR: {str(e)}", flush=True)
                raise
        
        # Process in batches for large PDFs
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size  # Ceiling division
        
        print(f">>> EMBEDDING: Processing {len(texts)} chunks in {total_batches} batches (batch_size={batch_size})", flush=True)
        
        for i in range(0, len(texts), batch_size):
            batch_num = i // batch_size + 1
            batch = texts[i:i + batch_size]
            
            print(f">>> EMBEDDING: Batch {batch_num}/{total_batches} ({len(batch)} chunks)", flush=True)
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to avoid rate limiting
                if batch_num < total_batches:
                    time.sleep(0.1)
                    
            except Exception as e:
                error_msg = str(e)
                print(f">>> EMBEDDING ERROR in batch {batch_num}/{total_batches}: {error_msg}", flush=True)
                
                # If it's a token limit error, suggest smaller batch size
                if "max_tokens_per_request" in error_msg.lower() or "token" in error_msg.lower():
                    print(f">>> EMBEDDING: Token limit exceeded. Try reducing batch_size (current: {batch_size})", flush=True)
                
                raise
        
        print(f">>> EMBEDDING: Successfully generated {len(all_embeddings)} embeddings", flush=True)
        return all_embeddings

    async def generate_embedding_async(self, text: str) -> List[float]:
        """Non-blocking single-text embedding (OpenAI async client)."""
        response = await self.async_client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    async def generate_embeddings_async(
        self, texts: List[str], batch_size: int = None
    ) -> List[List[float]]:
        """Non-blocking batched embeddings for ingest and retrieval."""
        if not texts:
            return []

        if batch_size is None:
            batch_size = self.default_batch_size

        if len(texts) <= batch_size:
            print(f">>> EMBEDDING (async): Processing {len(texts)} chunks in single batch", flush=True)
            response = await self.async_client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]

        all_embeddings: List[List[float]] = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        print(
            f">>> EMBEDDING (async): Processing {len(texts)} chunks in {total_batches} batches",
            flush=True,
        )

        for i in range(0, len(texts), batch_size):
            batch_num = i // batch_size + 1
            batch = texts[i : i + batch_size]
            print(
                f">>> EMBEDDING (async): Batch {batch_num}/{total_batches} ({len(batch)} chunks)",
                flush=True,
            )
            response = await self.async_client.embeddings.create(
                model=self.model,
                input=batch,
            )
            all_embeddings.extend(item.embedding for item in response.data)
            if batch_num < total_batches:
                await asyncio.sleep(0.1)

        print(
            f">>> EMBEDDING (async): Successfully generated {len(all_embeddings)} embeddings",
            flush=True,
        )
        return all_embeddings

