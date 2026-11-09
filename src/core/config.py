"""Конфигурация приложения на основе pydantic-settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки SPL, которые можно переопределить через .env или переменные окружения."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_name: str = "deepvk/GeRaCl-USER2-base"
    categories: list[str] = Field(
        default_factory=lambda: ["документы", "сроки", "общежитие", "экзамены", "оплата"]
    )
    db_path: str = "spl_logs.db"
    adversarial_model_path: str = "models/adversarial_detector.pkl"
    anomaly_model_path: str = "models/anomaly_detector.pkl"
    train_csv: str = "data/detector_train.csv"
    anomaly_contamination: float = 0.01
    log_level: str = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    """Вернуть единственный экземпляр настроек (ленивый синглтон)."""
    return Settings()
