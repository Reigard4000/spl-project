"""Кастомные исключения SPL."""


class SPLException(Exception):
    """Базовое исключение всех компонентов SPL."""


class DetectorError(SPLException):
    """Детектор не смог загрузить модель или обработать текст."""


class ModelError(SPLException):
    """Защищаемая ML-модель недоступна или вернула ошибку."""


class RepositoryError(SPLException):
    """Ошибка чтения или записи в хранилище логов."""
