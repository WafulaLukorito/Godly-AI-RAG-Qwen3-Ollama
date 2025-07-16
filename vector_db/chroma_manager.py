from typing import Optional
from langchain_chroma import Chroma
from data_processing.bible_loader import load_bible
from data_processing.text_splitter import ScriptureSplitter, should_split
from vector_db.embeddings import EmbeddingManager
from config.settings import config
import shutil
import os
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class ChromaManager:
    """Manages Chroma vector store lifecycle."""

    def __init__(self, embedding_manager: Optional[EmbeddingManager] = None):
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self._vector_store = None

    @property
    def vector_store(self):
        """Lazy-load the vector store."""
        if self._vector_store is None:
            self._vector_store = self._initialize_vector_store()
        return self._vector_store

    def _initialize_vector_store(self) -> Chroma:
        """Initialize or load existing vector store."""
        if self._store_exists():
            logger.info("Loading existing vector store")
            return self._load_existing_store()
        return self._create_new_store()

    def _store_exists(self) -> bool:
        """Check if Chroma store directory exists and is valid."""
        return (os.path.exists(config.chroma_path) and
                os.listdir(config.chroma_path))

    def _load_existing_store(self) -> Chroma:
        """Load persisted Chroma store."""
        try:
            return Chroma(
                persist_directory=config.chroma_path,
                embedding_function=self.embedding_manager.embedding_model
            )
        except Exception as e:
            logger.error(f"Failed to load existing store: {e}")
            raise RuntimeError("Vector store loading failed")

    def _create_new_store(self) -> Chroma:
        """Create new store from Bible verses."""
        logger.info("Creating new vector store")
        start_time = time.time()

        try:
            # Load and process documents
            verses = load_bible()
            if should_split(verses):
                verses = ScriptureSplitter().split_verses(verses)

            # Create store with progress callback
            def progress_callback(processed: int, total: int):
                if processed % 100 == 0:
                    logger.info(f"Embedding progress: {processed}/{total}")

            store = Chroma.from_documents(
                documents=verses,
                embedding=self.embedding_manager.embedding_model,
                persist_directory=config.chroma_path,
                collection_metadata={
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0",
                    "source": config.bible_xml
                }
            )

            logger.info(
                f"Created new store with {len(verses)} documents "
                f"in {time.time()-start_time:.2f}s"
            )
            return store

        except Exception as e:
            logger.error(f"Store creation failed: {e}", exc_info=True)
            raise RuntimeError("Vector store creation failed")

    def get_retriever(self, **kwargs):
        """Get configured retriever from vector store."""
        default_kwargs = {
            "search_type": "similarity_score_threshold",
            "search_kwargs": {
                "k": config.retrieval_k,
                "score_threshold": config.score_threshold
            }
        }
        return self.vector_store.as_retriever(
            **{**default_kwargs, **kwargs}
        )

    def reset_vector_store(self):
        """Force recreation of vector store."""

        if os.path.exists(config.chroma_path):
            shutil.rmtree(config.chroma_path)
        self._vector_store = None
