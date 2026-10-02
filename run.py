"""Launcher script for LangChain Agent Service."""

import uvicorn
from app.config import settings

if __name__ == "__main__":
    print(f"Starting {settings.APP_NAME} on http://127.0.0.1:{settings.PORT}")
    print(f"API Docs available at http://127.0.0.1:{settings.PORT}/docs")
    print(f"Interactive Web UI available at http://127.0.0.1:{settings.PORT}/")
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
