import os
import numpy as np
import faiss
from openai import AsyncOpenAI
from app.core.config import get_settings

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        api_key = settings.GEMINI_API_KEY.strip()
        _client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://dms.local",
                "X-OpenRouter-Title": "Medical RAG System",
            }
        )
    return _client


class SessionIndex:
    """In-memory FAISS index for a single analysis session. Resets per request."""

    def __init__(self):
        self.index: faiss.IndexFlatL2 | None = None
        self.chunks: list[str] = []
        self.dimension: int = 0

    def reset(self):
        self.index = None
        self.chunks = []
        self.dimension = 0

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        s = get_settings()
        size = s.CHUNK_SIZE
        overlap = s.CHUNK_OVERLAP
        if len(text) <= size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start += size - overlap
        return chunks

    async def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts using OpenRouter."""
        client = _get_client()
        model_name = get_settings().EMBEDDING_MODEL
        
        inputs = []
        for text in texts:
            inputs.append({
                "content": [{"type": "text", "text": text}]
            })
            
        resp = await client.embeddings.create(
            model=model_name,
            input=inputs,
            encoding_format="float"
        )
        embeddings = [item.embedding for item in resp.data]
        return np.array(embeddings, dtype=np.float32)

    async def _embed_query(self, text: str) -> np.ndarray:
        """Embed a single query."""
        client = _get_client()
        model_name = get_settings().EMBEDDING_MODEL
        
        resp = await client.embeddings.create(
            model=model_name,
            input=[{
                "content": [{"type": "text", "text": text}]
            }],
            encoding_format="float"
        )
        vec = resp.data[0].embedding
        return np.array([vec], dtype=np.float32)

    async def add_text(self, text: str) -> int:
        """Chunk, embed, and index text. Returns number of chunks added."""
        chunks = self._chunk_text(text)
        if not chunks:
            return 0

        embeddings = await self._embed(chunks)

        if self.index is None:
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)

        self.index.add(embeddings)
        self.chunks.extend(chunks)
        return len(chunks)

    async def search(self, query: str, top_k: int | None = None) -> list[str]:
        """Retrieve top-k most relevant chunks for a query."""
        if self.index is None or not self.chunks:
            return []

        k = min(top_k or get_settings().TOP_K, len(self.chunks))
        query_vec = await self._embed_query(query)
        _, indices = self.index.search(query_vec, k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results


# Global session index — reset per /analyze call
session_index = SessionIndex()
