"""Запуск API командой ``python -m src``."""

import uvicorn

from src.core.config import get_settings
from src.core.logging import setup_logging


def main() -> None:
    """Прочитать настройки и запустить uvicorn с фабрикой приложения."""
    settings = get_settings()
    setup_logging(settings.log_level)
    uvicorn.run(
        "src.presentation.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
