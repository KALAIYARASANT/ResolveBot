from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


class Settings:
    policies_path = Path(os.getenv("POLICIES_PATH", str(ROOT / "data/policies.md")))
    manifest_path = Path(os.getenv("MANIFEST_PATH", str(ROOT / "data/manifest.json")))
    db_path = Path(os.getenv("DB_PATH", str(ROOT / "data/shopsphere.db")))
    app_today = os.getenv("APP_TODAY", "2026-07-01")
    llm_provider = os.getenv("LLM_PROVIDER", "openai")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-6-luna")


@lru_cache
def get_settings() -> Settings:
    return Settings()
