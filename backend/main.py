"""
EduMind AI — Application Entry Point
Author: Mamani Kedarnath
Description: FastAPI application factory. Creates and configures
             the main app instance, registers all routers,
             sets up CORS, and starts the uvicorn server.
             This is the ONLY file that should be run directly.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from backend.core.config import settings


def create_app() -> FastAPI:
    """
    Application factory pattern.
    Creates and configures the FastAPI instance.
    Returns the configured app — makes testing easier.
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="RAG + LangGraph + Fine-Tuning Education Platform",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # --- CORS Middleware ---
    # Allows React frontend (localhost:5173) to call our API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Health Check Route ---
    @app.get("/api/v1/health")
    async def health_check():
        """
        Simple endpoint to verify the server is running.
        Used by deployment platforms to check app status.
        """
        return {
            "status": "healthy",
            "app": settings.app_name,
            "version": settings.app_version,
            "debug": settings.debug,
        }

    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
    )