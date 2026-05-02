"""
EduMind AI — Application Configuration
Author: Mamani Kedarnath
Description: Central config loader. Reads all environment variables
             from .env and exposes them as a typed Settings object.
             Every module imports from here — never reads .env directly.
             Follows Single Responsibility Principle.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    """
    All application settings loaded from .env file.
    Pydantic validates types automatically — wrong type = error at startup.
    """

    # --- App Settings ---
    app_name: str = Field(default="EduMind AI")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=True)
    port: int = Field(default=8000)

    # --- Groq API ---
    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="llama3-8b-8192")

    # --- Google OAuth ---
    google_client_id: str = Field(default="")
    google_client_secret: str = Field(default="")
    google_redirect_uri: str = Field(
        default="http://localhost:8000/api/v1/auth/callback"
    )

    # --- Database ---
    database_url: str = Field(default="sqlite:///./edumind.db")

    # --- ChromaDB ---
    chroma_persist_dir: str = Field(default="./data/vectorstore")

    # --- Security ---
    secret_key: str = Field(default="change-this-in-production")
    access_token_expire_minutes: int = Field(default=30)

    # --- File Upload ---
    max_file_size_mb: int = Field(default=50)
    upload_dir: str = Field(default="./data/uploads")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Returns cached Settings instance.
    lru_cache ensures .env is read only ONCE at startup — not on every request.
    Usage: from backend.core.config import get_settings
           settings = get_settings()
    """
    return Settings()


# Convenience instance for direct imports
settings = get_settings()