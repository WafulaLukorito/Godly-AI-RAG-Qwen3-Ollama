from pydantic import BaseSettings

class Settings(BaseSettings):
    data_path: str = "data/"
    bible_xml: str = "bible.xml"
    chroma_path: str = "bible_chroma_db"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "qwen3:8b"
    chunk_size: int = 300
    chunk_overlap: int = 50
    retrieval_k: int = 5
    score_threshold: float = 0.3

    class Config:
        env_file = ".env"


config = Settings()

LOGGING_LEVEL = "INFO"
LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
