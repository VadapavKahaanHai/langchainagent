"""Configuration for the local Gemini filing-review service."""

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Financial Filing Review"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEFAULT_PROVIDER: Literal["gemini"] = "gemini"
    GOOGLE_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-3.8-flash"
    AGENT_TEMPERATURE: float = 0.2
    AGENT_MAX_ITERATIONS: int = 10
    AGENT_VERBOSE: bool = True

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
