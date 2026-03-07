"""FastAPI Middleware configuration."""

from __future__ import annotations

import logging
import time
from typing import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def setup_middleware(app: FastAPI) -> None:
    """Attach middleware and exception handlers to the FastAPI app.

    Args:
        app: The FastAPI application instance.
    """
    # Cross-Origin Resource Sharing
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Note: Adding middleware using standard functions.
    # We define a custom request logging middleware directly on the app instance.
    @app.middleware("http")
    async def log_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Log incoming requests and their processing times."""
        start_time = time.time()

        # We don't log the body for file uploads as it gets messy
        logger.info(f"Incoming: {request.method} {request.url.path}")

        try:
            response = await call_next(request)
        except Exception as e:
            # Let the broad exception handler catch this below
            logger.error(f"Request failed: {e}")
            raise e

        process_time = time.time() - start_time
        logger.info(
            f"Completed: {request.method} {request.url.path} "
            f"- Status: {response.status_code} - Time: {process_time:.4f}s"
        )
        return response

    # Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch all unhandled exceptions and format them as JSON, logging the error.

        Args:
            request: The incoming request context.
            exc: The caught exception.

        Returns:
            A JSONResponse with status 500 containing human-readable error info.
        """
        # If it's a specific ValueError from our image processing logic, we could return 400
        if isinstance(exc, ValueError):
            logger.warning(
                f"Client error on {request.method} {request.url.path}: {exc}"
            )
            return JSONResponse(status_code=400, content={"detail": str(exc)})

        logger.exception("Unexpected global exception encountered.")
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred.", "error": str(exc)},
        )
