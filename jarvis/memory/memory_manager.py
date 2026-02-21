"""
jarvis.memory.memory_manager — Session memory + ChromaDB long-term vector memory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from jarvis.core.event_bus import AsyncEventBus
from jarvis.utils.config import MemoryConfig
from jarvis.utils.enums import MemoryType
from jarvis.utils.types import MemoryEntry

logger = logging.getLogger(__name__)


class SessionMemory:
    """In-process short-term memory for the current interaction session."""

    def __init__(self, max_turns: int = 50):
        self.max_turns = max_turns
        self.turns: list[dict[str, str]] = []
        self.context: dict[str, Any] = {}

    def add_turn(self, role: str, content: str):
        self.turns.append({"role": role, "content": content})
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def get_messages(self) -> list[dict[str, str]]:
        return list(self.turns)

    def set_context(self, key: str, value: Any):
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        return self.context.get(key, default)

    def clear(self):
        self.turns.clear()
        self.context.clear()


class JSONFileMemory:
    """Lightweight JSON-file fallback when ChromaDB is not installed."""

    def __init__(self, persist_dir: str):
        self._path = Path(persist_dir) / "memory.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict] = []
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                self._entries = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                self._entries = []

    def _save(self):
        self._path.write_text(json.dumps(self._entries, default=str), encoding="utf-8")

    def store(self, entry: MemoryEntry):
        # Replace if exists
        self._entries = [e for e in self._entries if e["id"] != entry.id]
        self._entries.append(
            {
                "id": entry.id,
                "content": entry.content,
                "type": entry.entry_type.value,
                "importance": entry.importance,
                "created_at": entry.created_at,
                "correlation_id": entry.source_correlation_id,
            }
        )
        self._save()

    def query(
        self, query_text: str, n_results: int = 5, min_relevance: float = 0.0
    ) -> list[MemoryEntry]:
        """Simple keyword/substring matching fallback."""
        keywords = set(query_text.lower().split())
        scored = []
        for e in self._entries:
            content_lower = e["content"].lower()
            matches = sum(1 for kw in keywords if kw in content_lower)
            if matches > 0:
                relevance = matches / max(len(keywords), 1)
                scored.append((relevance, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for relevance, e in scored[:n_results]:
            results.append(
                MemoryEntry(
                    id=e["id"],
                    content=e["content"],
                    entry_type=MemoryType(e.get("type", "conversation")),
                    importance=e.get("importance", 0.5),
                    created_at=e.get("created_at", 0.0),
                )
            )
        return results

    def delete(self, entry_id: str):
        self._entries = [e for e in self._entries if e["id"] != entry_id]
        self._save()

    def count(self) -> int:
        return len(self._entries)


class LongTermMemory:
    """
    Persistent long-term memory.
    Uses ChromaDB + sentence-transformers if available, otherwise falls back
    to a simple JSON file with keyword matching.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._collection = None
        self._embedder = None
        self._json_fallback: JSONFileMemory | None = None
        self._using_chromadb = False

    async def initialize(self):
        """Load ChromaDB and embedding model, or fall back to JSON."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._init_sync)

    def _init_sync(self):
        # Try ChromaDB first
        try:
            import chromadb

            client = chromadb.PersistentClient(path=self.config.persist_dir)
            self._collection = client.get_or_create_collection(
                name="jarvis_memory",
                metadata={"hnsw:space": "cosine"},
            )
            self._using_chromadb = True
            logger.info(
                "ChromaDB initialized at %s (%d entries)",
                self.config.persist_dir,
                self._collection.count(),
            )
        except Exception as e:
            logger.warning("ChromaDB not available: %s — using JSON fallback", e)
            self._collection = None

        # Try embedding model (only useful with ChromaDB)
        if self._using_chromadb:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(self.config.embedding_model)
                logger.info("Embedding model loaded: %s", self.config.embedding_model)
            except Exception as e:
                logger.warning("Embedding model not available: %s", e)

        # If ChromaDB failed, use JSON fallback
        if not self._using_chromadb:
            self._json_fallback = JSONFileMemory(self.config.persist_dir)
            logger.info(
                "📁 Using JSON file memory (%d entries) - install chromadb for vector search",
                self._json_fallback.count(),
            )

    def _embed(self, text: str) -> list[float]:
        if self._embedder is None:
            return []
        return self._embedder.encode(text).tolist()

    def store(self, entry: MemoryEntry):
        if self._json_fallback:
            self._json_fallback.store(entry)
            return
        if self._collection is None:
            return
        embedding = (
            self._embed(entry.content) if not entry.embedding else entry.embedding
        )
        self._collection.upsert(
            ids=[entry.id],
            embeddings=[embedding] if embedding else None,
            documents=[entry.content],
            metadatas=[
                {
                    "type": entry.entry_type.value,
                    "importance": entry.importance,
                    "created_at": entry.created_at,
                    "source_correlation_id": entry.source_correlation_id,
                }
            ],
        )

    def query(
        self, query_text: str, n_results: int = 5, min_relevance: float = 0.0
    ) -> list[MemoryEntry]:
        if self._json_fallback:
            return self._json_fallback.query(query_text, n_results, min_relevance)
        if self._collection is None:
            return []
        embedding = self._embed(query_text)
        if not embedding:
            return []

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, self.config.max_context_entries),
            include=["documents", "metadatas", "distances"],
        )

        entries = []
        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i]
            relevance = 1.0 - distance  # cosine distance → similarity
            if relevance < min_relevance:
                continue
            meta = results["metadatas"][0][i]
            entries.append(
                MemoryEntry(
                    id=results["ids"][0][i],
                    content=doc,
                    entry_type=MemoryType(meta.get("type", "conversation")),
                    importance=meta.get("importance", 0.5),
                    created_at=meta.get("created_at", 0.0),
                )
            )
        return entries

    def delete(self, entry_id: str):
        if self._json_fallback:
            self._json_fallback.delete(entry_id)
        elif self._collection:
            self._collection.delete(ids=[entry_id])

    def count(self) -> int:
        if self._json_fallback:
            return self._json_fallback.count()
        return self._collection.count() if self._collection else 0


class MemoryManager:
    """
    Unified memory manager combining session and long-term memory.
    Provides the interface used by the orchestrator and cognition modules.
    """

    def __init__(self, config: MemoryConfig, event_bus: AsyncEventBus):
        self.config = config
        self.event_bus = event_bus
        self.session = SessionMemory()
        self.long_term = LongTermMemory(config)

    async def initialize(self):
        await self.long_term.initialize()
        logger.info(
            "Memory manager initialized (LTM entries: %d)", self.long_term.count()
        )

    def add_user_turn(self, text: str):
        self.session.add_turn("user", text)

    def add_assistant_turn(self, text: str):
        self.session.add_turn("assistant", text)

    def get_conversation_messages(self) -> list[dict[str, str]]:
        return self.session.get_messages()

    def retrieve_relevant(self, query: str, n: int = 5) -> list[MemoryEntry]:
        return self.long_term.query(
            query, n_results=n, min_relevance=self.config.relevance_threshold
        )

    async def store_fact(
        self, content: str, importance: float = 0.7, correlation_id: str = ""
    ):
        entry = MemoryEntry(
            content=content,
            entry_type=MemoryType.FACT,
            importance=importance,
            source_correlation_id=correlation_id,
        )
        await asyncio.get_event_loop().run_in_executor(
            None, self.long_term.store, entry
        )

    async def store_task_outcome(
        self, content: str, importance: float = 0.8, correlation_id: str = ""
    ):
        entry = MemoryEntry(
            content=content,
            entry_type=MemoryType.TASK_OUTCOME,
            importance=importance,
            source_correlation_id=correlation_id,
        )
        await asyncio.get_event_loop().run_in_executor(
            None, self.long_term.store, entry
        )

    async def store_preference(self, content: str, correlation_id: str = ""):
        entry = MemoryEntry(
            content=content,
            entry_type=MemoryType.PREFERENCE,
            importance=0.9,
            source_correlation_id=correlation_id,
        )
        await asyncio.get_event_loop().run_in_executor(
            None, self.long_term.store, entry
        )

    async def store_interaction(self, event: dict):
        """Store interaction outcome in long-term memory."""
        content = event.get("summary", event.get("response", ""))
        if content:
            await self.store_task_outcome(content)

    def format_memory_context(self, query: str) -> str:
        """Format retrieved memories for LLM context injection."""
        memories = self.retrieve_relevant(query)
        if not memories:
            return ""
        lines = ["## Retrieved Memories"]
        for m in memories:
            lines.append(f"- [{m.entry_type.value.upper()}] {m.content}")
        return "\n".join(lines)
