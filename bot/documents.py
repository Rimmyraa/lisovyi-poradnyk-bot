"""Завантаження архіву і документів, які звірі надсилають у Telegram."""
from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .classify import classify_document
from .extract import SUPPORTED, UnsupportedFormat, extract_text, make_title, normalize
from .forest_map import FOREST, ForestMap
from .llm import LLM
from .storage import Document, PendingDoc, Storage

log = logging.getLogger(__name__)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class UploadResult:
    status: str                       # saved | duplicate | outdated | need_place | empty | unsupported
    document: Document | None = None
    pending: PendingDoc | None = None
    title: str = ""
    replaced: list[Document] = field(default_factory=list)  # старі версії, які замінив цей документ


class DocumentService:
    def __init__(self, storage: Storage, fmap: ForestMap, llm: LLM | None, tessdata: str = ""):
        self.storage, self.fmap, self.llm, self.tessdata = storage, fmap, llm, tessdata

    async def _extract(self, path: Path) -> tuple[str, bool]:
        text, ocr = await asyncio.to_thread(extract_text, path, self.tessdata)
        return normalize(text), ocr

    # ---- архів ----
    async def ingest_archive(self, folder: Path) -> int:
        if not folder.exists():
            log.warning("Теки архіву немає: %s", folder)
            return 0
        added = 0
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in SUPPORTED:
                continue
            data = path.read_bytes()
            sha = sha256_bytes(data)
            if self.storage.has_sha(sha):
                continue
            try:
                text, ocr = await self._extract(path)
            except Exception as e:
                log.error("Не вдалося прочитати %s: %s", path.name, e)
                continue
            if not text:
                log.warning("Порожній документ в архіві: %s", path.name)
                continue
            title = make_title(path.name, text)
            places = await classify_document(title, text, "", self.fmap, self.llm)
            if not places:
                log.warning("Не вдалося визначити область для %s — вважаю порадою для всього лісу", path.name)
                places = [FOREST]
            mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
            doc_id = self.storage.add_document(title=title, filename=path.name, source="archive", places=places,
                                               text=text, sha256=sha, sent_at=mtime, ocr=ocr)
            self.storage.supersede_versions(self.storage.get(doc_id))
            added += 1
            log.info("Архів: %s → %s%s", title, places, " (OCR)" if ocr else "")
        return added

    def load_map(self) -> None:
        """Оновлює відповідність рукавичок і частин лісу з документа «Карта лісу»."""
        for d in self.storage.all_documents():
            if "карта" in d.title.lower():
                fm = ForestMap.from_map_text(d.text)
                self.fmap.gloves = fm.gloves
                log.info("Карта лісу прочитана з «%s»: %s", d.title, fm.gloves)
                return
        log.info("Документа «Карта лісу» немає — використовую відому відповідність")

    # ---- Telegram ----
    async def handle_upload(self, *, user_id: int, chat_id: int, sender_name: str, filename: str, data: bytes,
                            caption: str = "", sent_at: str | None = None, tmp_dir: Path) -> UploadResult:
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED:
            return UploadResult("unsupported", title=filename)
        sha = sha256_bytes(data)
        existing = self.storage.has_sha(sha)
        if existing:
            if existing.superseded_by:
                current = self.storage.current_version(existing)
                return UploadResult("outdated", document=current, title=existing.title)
            return UploadResult("duplicate", document=existing, title=existing.title)
        pend = self.storage.pending_doc_by_sha(user_id, sha)
        if pend:
            return UploadResult("need_place", pending=pend, title=pend.title)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        path = tmp_dir / f"{sha[:16]}{ext}"
        path.write_bytes(data)
        try:
            text, ocr = await self._extract(path)
        except UnsupportedFormat:
            return UploadResult("unsupported", title=filename)
        except Exception as e:
            log.error("Помилка читання %s: %s", filename, e)
            return UploadResult("empty", title=filename)
        finally:
            path.unlink(missing_ok=True)
        if len(text) < 20:
            return UploadResult("empty", title=filename)
        title = make_title(filename, text)
        sent_at = sent_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        places = await classify_document(title, text, caption, self.fmap, self.llm)
        if not places:
            pid = self.storage.add_pending_doc(user_id=user_id, chat_id=chat_id, title=title, filename=filename,
                                               sender_name=sender_name, sent_at=sent_at, text=text, ocr=ocr,
                                               sha256=sha)
            return UploadResult("need_place", pending=self.storage.get_pending_doc(pid), title=title)
        doc_id = self.storage.add_document(title=title, filename=filename, source="telegram", places=places,
                                           text=text, sha256=sha, sent_at=sent_at, sender_id=user_id,
                                           sender_name=sender_name, ocr=ocr)
        doc = self.storage.get(doc_id)
        replaced = self.storage.supersede_versions(doc)
        return UploadResult("saved", document=doc, title=title, replaced=replaced)

    def resolve_pending(self, pending: PendingDoc, places: list[str]) -> tuple[Document, list[Document]]:
        """Зберігає документ після уточнення місця. Повертає (документ, замінені старі версії)."""
        existing = self.storage.has_sha(pending.sha256)
        if existing:
            self.storage.delete_pending_doc(pending.id)
            return self.storage.current_version(existing), []
        doc_id = self.storage.add_document(title=pending.title, filename=pending.filename, source="telegram",
                                           places=places, text=pending.text, sha256=pending.sha256,
                                           sent_at=pending.sent_at, sender_id=pending.user_id,
                                           sender_name=pending.sender_name, ocr=pending.ocr)
        self.storage.delete_pending_doc(pending.id)
        doc = self.storage.get(doc_id)
        return doc, self.storage.supersede_versions(doc)


async def sync_drive(folder_id: str, api_key: str, dest: Path) -> bool:
    """Завантажує файли публічної теки Google Диска (потрібен API-ключ Google). True — якщо вдалося."""
    import aiohttp

    dest.mkdir(parents=True, exist_ok=True)
    base = "https://www.googleapis.com/drive/v3/files"
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as s:
            files, token = [], None
            while True:
                params = {"q": f"'{folder_id}' in parents and trashed=false", "key": api_key,
                          "fields": "nextPageToken, files(id, name, mimeType)", "pageSize": "1000"}
                if token:
                    params["pageToken"] = token
                async with s.get(base, params=params) as r:
                    if r.status != 200:
                        log.warning("Google Drive: %s %s", r.status, await r.text())
                        return False
                    js = await r.json()
                files += js.get("files", [])
                token = js.get("nextPageToken")
                if not token:
                    break
            for f in files:
                name, mime = f["name"], f["mimeType"]
                if mime == "application/vnd.google-apps.document":
                    url, params, name = f"{base}/{f['id']}/export", {"mimeType": "text/plain", "key": api_key}, name + ".txt"
                elif Path(name).suffix.lower() in SUPPORTED:
                    url, params = f"{base}/{f['id']}", {"alt": "media", "key": api_key}
                else:
                    continue
                async with s.get(url, params=params) as r:
                    if r.status == 200:
                        (dest / name).write_bytes(await r.read())
            log.info("Google Drive: завантажено %d файлів", len(list(dest.iterdir())))
            return True
    except Exception as e:
        log.warning("Синхронізація з Google Drive не вдалася: %s", e)
        return False
