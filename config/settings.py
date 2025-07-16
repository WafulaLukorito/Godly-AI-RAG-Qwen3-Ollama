from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # File system settings
    data_path: str = "data/"
    bible_xml: str = "bible.xml"
    chroma_path: str = "bible_chroma_db"

    # Model settings
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "qwen3:8b"

    # Text processing settings
    chunk_size: int = 300
    chunk_overlap: int = 50

    # Retrieval settings
    retrieval_k: int = 5
    score_threshold: float = 0.3

    # Logging settings
    logging_level: Literal['DEBUG', 'INFO',
                           'WARNING', 'ERROR', 'CRITICAL'] = "INFO"
    logging_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging_file: str = "logs/biblical_counselor.log"

    # Performance settings
    timeout: int = 300  # seconds
    num_gpu: int = 1  # 0 for CPU-only, 1 for GPU acceleration

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = False


# Singleton configuration instance
config = Settings()


def get_logger_config() -> dict:
    """Return logging configuration dictionary."""
    return {
        'level': config.logging_level,
        'format': config.logging_format,
        'file': config.logging_file,
        'rotation': "5 MB",
        'retention': "100 days"
    }
