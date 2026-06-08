"""Environment-driven configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    db_path: Path = Field(default=PROJECT_ROOT / "data" / "lemon.db", alias="LEMON_DB_PATH")

    lm_studio_base_url: str = Field(
        default="http://localhost:1234/v1", alias="LM_STUDIO_BASE_URL"
    )
    lm_studio_api_key: str = Field(default="lm-studio", alias="LM_STUDIO_API_KEY")
    lm_studio_logs_dir: Path | None = Field(default=None, alias="LM_STUDIO_LOGS_DIR")

    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )

    classifier_llm_provider: Literal["none", "lm_studio", "openrouter"] = Field(
        default="none", alias="CLASSIFIER_LLM_PROVIDER"
    )
    classifier_llm_model: str = Field(
        default="meta-llama/llama-3.1-8b-instruct", alias="CLASSIFIER_LLM_MODEL"
    )

    default_token_encoding: str = Field(default="cl100k_base", alias="DEFAULT_TOKEN_ENCODING")

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.db_path}"


settings = Settings()
