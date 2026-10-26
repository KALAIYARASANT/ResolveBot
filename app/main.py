from fastapi import FastAPI
from app.api.router import router

app = FastAPI(title="ResolveBot", version="1.0.0", docs_url=None, redoc_url=None, openapi_url=None)
app.include_router(router)
