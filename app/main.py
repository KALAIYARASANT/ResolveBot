import logging

from fastapi import FastAPI
from app.api.router import router
from app.core.config import get_settings

settings = get_settings()
settings.validate()  # unsupported provider or empty model: stop at startup with a clear message
if not settings.openai_api_key:
    logging.getLogger("uvicorn.error").warning(
        "OPENAI_API_KEY is not set: only the limited deterministic fallback is available."
    )

app = FastAPI(title="ResolveBot", version="1.0.0", docs_url=None, redoc_url=None, openapi_url=None)
app.include_router(router)
