"""Centralized configuration for the benchmark framework."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # LLM
    llm_provider: str = Field(default="anthropic", description="anthropic or openai")
    llm_model: str = Field(default="claude-sonnet-4-20250514")
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Database
    database_url: str = Field(default=f"sqlite:///{ROOT_DIR / 'data' / 'db' / 'benchmark.db'}")

    # Vector DB
    chroma_persist_dir: str = Field(default=str(ROOT_DIR / "data" / "vectordb"))

    # Paths
    data_dir: Path = ROOT_DIR / "data"
    csv_dir: Path = ROOT_DIR / "data" / "csv"
    tasks_dir: Path = ROOT_DIR / "tasks"
    results_dir: Path = ROOT_DIR / "results"
    plots_dir: Path = ROOT_DIR / "plots"

    # Logging
    log_level: str = "INFO"

    model_config = {"env_file": str(ROOT_DIR / ".env"), "env_file_encoding": "utf-8"}


settings = Settings()
