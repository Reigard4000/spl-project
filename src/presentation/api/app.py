"""Сборка FastAPI-приложения SPL."""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.core.config import get_settings
from src.core.exceptions import SPLException
from src.core.logging import setup_logging
from src.infrastructure.db.sqlite_repo import SQLiteLogRepository
from src.presentation.api.deps import get_log_repo
from src.presentation.api.routes import router

logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Инициализировать базу данных при старте приложения."""
    repo_provider: Callable[[], SQLiteLogRepository] = app.dependency_overrides.get(
        get_log_repo, get_log_repo
    )
    repository = repo_provider()
    repository.init_db()
    logger.info("SPL API запущен")
    yield
    logger.info("SPL API остановлен")


def create_app() -> FastAPI:
    """Создать и настроить FastAPI-приложение."""
    settings = get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title="SPL — Security & Protection Layer",
        description="Слой защиты ML-модели от adversarial-атак.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    _register_exception_handlers(app)
    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """Зарегистрировать обработчики кастомных и валидационных ошибок."""

    @app.exception_handler(SPLException)
    async def spl_exception_handler(
        _request: Request,
        exc: SPLException,
    ) -> JSONResponse:
        logger.error("Ошибка SPL: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": exc.__class__.__name__,
                "detail": str(exc),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "ValidationError",
                "detail": jsonable_encoder(exc.errors()),
            },
        )

    @app.exception_handler(ValidationError)
    async def pydantic_validation_handler(
        _request: Request,
        exc: ValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "ValidationError",
                "detail": jsonable_encoder(exc.errors()),
            },
        )
