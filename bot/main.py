"""Точка входу: завантажує архів, читає карту лісу і запускає Telegram-бота."""
from __future__ import annotations

import asyncio
import logging
import sys

from .config import Config
from .documents import DocumentService, sync_drive
from .forest_map import ForestMap
from .llm import GroqLLM
from .qa import QA
from .storage import Storage

log = logging.getLogger("lisovyi")


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    cfg = Config.from_env()
    if not cfg.telegram_token or not cfg.groq_api_key:
        log.error("Потрібні змінні оточення TELEGRAM_BOT_TOKEN і GROQ_API_KEY (див. .env.example)")
        sys.exit(1)

    storage = Storage(cfg.data_dir / "bot.db")
    fmap = ForestMap()
    llm = GroqLLM(cfg.groq_api_key, cfg.groq_models, cfg.groq_fast_models)
    await llm.check_models()
    docs = DocumentService(storage, fmap, llm, cfg.tessdata_prefix)

    archive = cfg.archive_dir
    if cfg.gdrive_folder_id and cfg.google_api_key:
        drive_dir = cfg.data_dir / "drive_archive"
        if await sync_drive(cfg.gdrive_folder_id, cfg.google_api_key, drive_dir):
            archive = drive_dir
    log.info("Завантажую «Лісовий архів» з %s (перший запуск із OCR може тривати кілька хвилин)…", archive)
    added = await docs.ingest_archive(archive)
    log.info("Додано нових документів: %d; усього в пам'яті: %d", added, storage.count())
    docs.load_map()

    qa = QA(storage, fmap, llm, cfg.timezone, cfg.max_context_chars)

    from aiogram import Bot, Dispatcher
    from aiogram.types import BotCommand

    from .handlers import build_router

    bot = Bot(cfg.telegram_token)
    dp = Dispatcher()
    dp.include_router(build_router(qa, docs, fmap, cfg.data_dir / "tmp", cfg.max_file_mb))
    await bot.set_my_commands([
        BotCommand(command="start", description="Що я вмію"),
        BotCommand(command="places", description="Рукавички і частини лісу"),
        BotCommand(command="help", description="Довідка"),
    ])
    log.info("Бот запущено")
    try:
        await dp.start_polling(bot)
    finally:
        await llm.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
