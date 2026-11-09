"""Единая настройка логирования для API, панели и скриптов."""

import logging

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """Настроить корневой логгер, не создавая дублирующихся обработчиков."""
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=numeric_level,
            format=LOG_FORMAT,
            datefmt=DATE_FORMAT,
        )
    root_logger.setLevel(numeric_level)
