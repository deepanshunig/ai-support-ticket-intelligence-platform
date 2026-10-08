"""Validated configuration read from TICKET_* environment variables and .env."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TICKET_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(
        default="AI Support Ticket Intelligence Platform", min_length=1, max_length=100
    )
    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    database_url: SecretStr | None = None
    artifact_dir: Path = Path("artifacts/baselines")
    team_mapping_path: Path = Path("configs/team_mapping.json")
    reviewer_token: SecretStr | None = None
    reviewer_id: str = Field(default="demo-reviewer", min_length=1, max_length=100)
    webhook_url: SecretStr | None = None
    webhook_secret: SecretStr | None = None
    policy_path: Path | None = Path("artifacts/routing/policy.json")
