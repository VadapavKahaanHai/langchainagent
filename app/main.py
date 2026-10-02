"""FastAPI Application Entrypoint for LangChain Agent Service."""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.api.endpoints import router as agent_router
from app.models.schemas import HealthResponse

# Logging configuration
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("langchain_agent_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown routines."""
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION} in {settings.ENVIRONMENT} mode...")
    logger.info(f"Default LLM Provider: {settings.DEFAULT_PROVIDER}")
    yield
    logger.info("🛑 Shutting down LangChain Agent Service...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Exploratory financial filing review using Gemini and a local text classifier.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Static files for web UI
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Root Web UI
@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the interactive web console."""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "LangChain Agent Service API is active. Visit /docs for Swagger documentation."}


# Health check
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring."""
    return HealthResponse(
        status="ok",
        app=settings.APP_NAME,
        version=settings.APP_VERSION,
        default_provider=settings.DEFAULT_PROVIDER,
    )


# Register API routers
app.include_router(agent_router)
