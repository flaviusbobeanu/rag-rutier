"""Project settings, loaded from environment variables and the .env file."""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All configurable values in one place. Defaults match .env.example."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    llm_provider: str = "anthropic"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: str = "medium"  # low | medium | high | xhigh | max
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    embedding_model: str = "BAAI/bge-m3"
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    # Each experiment variant (chunking, embedding model) gets its own chunks file and collection.
    chunks_file: str = "chunks.jsonl"
    collection_name: str = "laws"
    retrieval_mode: str = "vector"  # vector | bm25 | hybrid
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Full path to tesseract.exe, only needed if Tesseract is not on PATH (empty = look on PATH)
    tesseract_cmd: str = ""

    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"

    @field_validator("chroma_dir", "raw_dir", "processed_dir")
    @classmethod
    def relative_to_project(cls, path: Path) -> Path:
        """A relative path in .env (e.g. CHROMA_DIR=chroma_db) means relative to the project folder,
        not to wherever the command happens to be run from."""
        return path if path.is_absolute() else PROJECT_ROOT / path


settings = Settings()
