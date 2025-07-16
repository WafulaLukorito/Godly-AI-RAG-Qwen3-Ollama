from typing import Optional
from langchain_ollama import OllamaEmbeddings
from config.settings import config
import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)


class EmbeddingModelType(Enum):
    """Supported embedding model types."""
    OLLAMA = auto()
    # Add other types like OPENAI, HUGGINGFACE later


class EmbeddingManager:
    """Handles embedding model initialisation and operations."""  # UK English spelling

    def __init__(self, model_type: EmbeddingModelType = EmbeddingModelType.OLLAMA):
        self.model_type = model_type
        self._embedding_model = None

    @property
    def embedding_model(self):
        """Lazy-load the embedding model."""
        if self._embedding_model is None:
            self._embedding_model = self._initialize_model()
        return self._embedding_model

    def _initialize_model(self) -> OllamaEmbeddings:
        """Initialise the specified embedding model."""  # UK English spelling
        try:
            # UK English spelling
            logger.info(
                f"Initialising {self.model_type.name} embedding model with model: {config.embedding_model}")

            if self.model_type == EmbeddingModelType.OLLAMA:
                # Remove num_gpu and timeout.
                # Ollama's GPU usage is configured at the Ollama server level,
                # not via this client library's constructor.
                # Timeouts are also not directly configurable here for OllamaEmbeddings.
                return OllamaEmbeddings(
                    model=config.embedding_model,
                    # If you need to specify base_url for Ollama server, you can add it here:
                    # base_url="http://localhost:11434" # Default Ollama URL
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        except Exception as e:
            # UK English spelling
            logger.error(
                f"Failed to initialise embedding model: {e}", exc_info=True)
            # UK English spelling
            raise RuntimeError(f"Embedding initialisation failed: {e}")

    def get_embedding_function(self):
        """Get the embedding function for vector stores."""
        # For OllamaEmbeddings, embed_documents is the correct method
        # for batch embedding, and embed_query for single string.
        # ChromaDB expects a function that can take a list of texts.
        return self.embedding_model.embed_documents

    def test_connection(self) -> bool:
        """Verify the embedding model is operational."""
        try:
            test_text = "connection test"
            # Use embed_query for single string test
            self.embedding_model.embed_query(test_text)
            logger.info("Embedding model connection test successful.")
            return True
        except Exception as e:
            logger.warning(
                f"Embedding model connection test failed: {e}", exc_info=True)
            return False
