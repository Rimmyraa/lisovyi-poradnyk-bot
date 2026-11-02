"""Налаштування з змінних оточення."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass
class Config:
    telegram_token: str = ""
    groq_api_key: str = ""
    # Основна модель для відповідей і класифікації, далі — запасні (якщо ліміт або модель зникла).
    groq_models: list[str] = field(default_factory=list)
    # Легка модель для розбору питань.
    groq_fast_models: list[str] = field(default_factory=list)
    data_dir: Path = Path("/data")
    archive_dir: Path = Path("/app/data/archive")
    gdrive_folder_id: str = ""
    google_api_key: str = ""
    timezone: str = "Europe/Kyiv"
    tessdata_prefix: str = ""
    max_context_chars: int = 9000
    max_file_mb: int = 20

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            telegram_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            groq_models=_list(os.getenv(
                "GROQ_MODELS",
                "llama-3.3-70b-versatile,meta-llama/llama-4-scout-17b-16e-instruct,"
                "openai/gpt-oss-120b,qwen/qwen3-32b",
            )),
            groq_fast_models=_list(os.getenv(
                "GROQ_FAST_MODELS",
                "llama-3.1-8b-instant,meta-llama/llama-4-scout-17b-16e-instruct,llama-3.3-70b-versatile",
            )),
            data_dir=Path(os.getenv("DATA_DIR", "/data")),
            archive_dir=Path(os.getenv("ARCHIVE_DIR", "/app/data/archive")),
            gdrive_folder_id=os.getenv("GDRIVE_FOLDER_ID", ""),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            timezone=os.getenv("TZ", "Europe/Kyiv"),
            tessdata_prefix=os.getenv("TESSDATA_PREFIX", ""),
            max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "9000")),
        )
