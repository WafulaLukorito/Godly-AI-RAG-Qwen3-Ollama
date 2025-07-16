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
    """Handles embedding model initialization and operations."""
    
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
        """Initialize the specified embedding model."""
        try:
            logger.info(f"Initializing {self.model_type.name} embedding model")
            
            if self.model_type == EmbeddingModelType.OLLAMA:
                return OllamaEmbeddings(
                    model=config.embedding_model,
                    num_gpu=1,
                    timeout=300
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}", exc_info=True)
            raise RuntimeError(f"Embedding initialization failed: {e}")

    def get_embedding_function(self):
        """Get the embedding function for vector stores."""
        return self.embedding_model.embed_documents

    def test_connection(self) -> bool:
        """Verify the embedding model is operational."""
        try:
            test_text = "connection test"
            self.embedding_model.embed_query(test_text)
            return True
        except Exception as e:
            logger.warning(f"Embedding test failed: {e}")
            return False
        