from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from config.settings import config
import logging
import time
import threading

logger = logging.getLogger(__name__)

class ScriptureSplitter:
    """Handles verse-aware text splitting with progress tracking."""
    
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! "],
            length_function=len,
            keep_separator=True,
            is_separator_regex=False
        )
        self._updates_active = False

    def _progress_thread(self, start_time: float):
        """Background thread showing progress updates."""
        while self._updates_active:
            elapsed = time.time() - start_time
            logger.info(f"Splitting in progress ({elapsed:.1f}s)")
            time.sleep(5)

    def split_verses(self, documents: List[Document]) -> List[Document]:
        """Split documents with progress tracking.
        
        Args:
            documents: List of Bible verse Documents
            
        Returns:
            List of split Documents with preserved metadata
        """
        start_time = time.time()
        self._updates_active = True
        
        # Start progress thread
        progress_thread = threading.Thread(
            target=self._progress_thread,
            args=(start_time,),
            daemon=True
        )
        progress_thread.start()

        try:
            chunks = self.splitter.split_documents(documents)
            
            # Preserve original metadata in chunks
            for chunk in chunks:
                if not chunk.metadata.get("is_chunk"):
                    chunk.metadata["original_content"] = chunk.page_content
                    chunk.metadata["is_chunk"] = True

            return chunks
            
        finally:
            self._updates_active = False
            progress_thread.join()
            logger.info(
                f"Split {len(documents)} verses into {len(chunks)} chunks "
                f"in {time.time()-start_time:.2f}s"
            )

def should_split(documents: List[Document]) -> bool:
    """Check if any verses exceed chunk size."""
    return any(len(doc.page_content) > config.chunk_size for doc in documents)