from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

SUPPORTED_PROVIDERS = ("openai",)


class ConfigurationError(RuntimeError):
    """The service is misconfigured; fail with a clear message instead of guessing."""


class Settings:
    policies_path = Path(os.getenv("POLICIES_PATH", str(ROOT / "data/policies.md")))
    manifest_path = Path(os.getenv("MANIFEST_PATH", str(ROOT / "data/manifest.json")))
    db_path = Path(os.getenv("DB_PATH", str(ROOT / "data/shopsphere.db")))
    app_today = os.getenv("APP_TODAY", "2026-07-01")
    llm_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    # Any chat model that supports function calling and temperature 0.
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    # Only for reasoning models that reject temperature 0 unless reasoning is disabled (for example "none").
    openai_reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "").strip()

    def validate(self) -> None:
        if self.llm_provider not in SUPPORTED_PROVIDERS:
            raise ConfigurationError(
                f"LLM_PROVIDER={self.llm_provider!r} is not supported; this build supports only: {', '.join(SUPPORTED_PROVIDERS)}."
            )
        if not self.openai_model:
            raise ConfigurationError("OPENAI_MODEL must not be empty.")


@lru_cache
def get_settings() -> Settings:
    return Settings()
