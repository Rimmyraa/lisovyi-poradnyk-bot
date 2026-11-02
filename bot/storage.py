"""SQLite: документи, документи, що чекають уточнення місця, і відкладені питання."""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    filename TEXT NOT NULL,
    source TEXT NOT NULL,              -- 'archive' | 'telegram'
    places TEXT NOT NULL,              -- JSON-список областей: ["весь ліс"] / ["хвойний ліс"] / ["Під дубом"]
    sender_id INTEGER,
    sender_name TEXT,
    sent_at TEXT NOT NULL,             -- ISO 8601, UTC
    text TEXT NOT NULL,
    ocr INTEGER NOT NULL DEFAULT 0,
    sha256 TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS pending_docs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    filename TEXT NOT NULL,
    sender_name TEXT,
    sent_at TEXT NOT NULL,
    text TEXT NOT NULL,
    ocr INTEGER NOT NULL DEFAULT 0,
    sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pending_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS user_context (
    user_id INTEGER PRIMARY KEY,
    last_question TEXT,
    last_places TEXT,
    updated_at TEXT
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Document:
    id: int
    title: str
    filename: str
    source: str
    places: list[str]
    sender_id: int | None
    sender_name: str | None
    sent_at: str
    text: str
    ocr: bool = False
    sha256: str = ""
    superseded_by: int | None = None


def title_key(title: str) -> str:
    t = title.lower().replace("’", "'").replace("ʼ", "'")
    t = re.sub(r"\.(pdf|docx|txt)$", "", t)
    t = re.sub(r"\(\d+\)$|\bкопія\b|\bcopy\b|\bv?\d+(\.\d+)*$", "", t.strip())  # «назва (1)», «назва v2»
    return re.sub(r"[^a-zа-яіїєґ0-9]+", " ", t).strip()


@dataclass
class PendingDoc:
    id: int
    user_id: int
    chat_id: int
    title: str
    filename: str
    sender_name: str | None
    sent_at: str
    text: str
    ocr: bool
    sha256: str


class Storage:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(documents)")}
        if "superseded_by" not in cols:  # міграція старої бази
            self.db.execute("ALTER TABLE documents ADD COLUMN superseded_by INTEGER")
        self.db.commit()

    # ---- documents ----
    def has_sha(self, sha: str) -> Document | None:
        row = self.db.execute("SELECT * FROM documents WHERE sha256=?", (sha,)).fetchone()
        return self._doc(row) if row else None

    def add_document(self, *, title: str, filename: str, source: str, places: list[str], text: str,
                     sha256: str, sent_at: str | None = None, sender_id: int | None = None,
                     sender_name: str | None = None, ocr: bool = False) -> int:
        cur = self.db.execute(
            "INSERT INTO documents(title, filename, source, places, sender_id, sender_name, sent_at, text, ocr, sha256)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (title, filename, source, json.dumps(places, ensure_ascii=False), sender_id, sender_name,
             sent_at or now_iso(), text, int(ocr), sha256),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def all_documents(self, include_superseded: bool = False) -> list[Document]:
        where = "" if include_superseded else " WHERE superseded_by IS NULL"
        return [self._doc(r) for r in self.db.execute(f"SELECT * FROM documents{where} ORDER BY id")]

    def get(self, doc_id: int) -> Document | None:
        r = self.db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return self._doc(r) if r else None

    def current_version(self, doc: Document) -> Document:
        """Для заміненого документа повертає чинну версію."""
        while doc.superseded_by:
            nxt = self.get(doc.superseded_by)
            if not nxt:
                break
            doc = nxt
        return doc

    def supersede_versions(self, new: Document) -> list[Document]:
        """Новіша версія того самого документа для того самого місця замінює старі.

        «Той самий документ» = однакова назва (без урахування регістру й розділових знаків)
        і однаковий набір місць.
        """
        replaced = []
        for d in self.all_documents():
            if d.id != new.id and title_key(d.title) == title_key(new.title) and set(d.places) == set(new.places):
                self.db.execute("UPDATE documents SET superseded_by=? WHERE id=?", (new.id, d.id))
                replaced.append(d)
        self.db.commit()
        return replaced

    def count(self, source: str | None = None) -> int:
        q = "SELECT COUNT(*) FROM documents WHERE superseded_by IS NULL"
        if source:
            return self.db.execute(q + " AND source=?", (source,)).fetchone()[0]
        return self.db.execute(q).fetchone()[0]

    @staticmethod
    def _doc(r: sqlite3.Row) -> Document:
        return Document(id=r["id"], title=r["title"], filename=r["filename"], source=r["source"],
                        places=json.loads(r["places"]), sender_id=r["sender_id"], sender_name=r["sender_name"],
                        sent_at=r["sent_at"], text=r["text"], ocr=bool(r["ocr"]), sha256=r["sha256"],
                        superseded_by=r["superseded_by"])

    # ---- pending docs ----
    def add_pending_doc(self, *, user_id: int, chat_id: int, title: str, filename: str, sender_name: str | None,
                        sent_at: str, text: str, ocr: bool, sha256: str) -> int:
        cur = self.db.execute(
            "INSERT INTO pending_docs(user_id, chat_id, title, filename, sender_name, sent_at, text, ocr, sha256)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (user_id, chat_id, title, filename, sender_name, sent_at, text, int(ocr), sha256),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def get_pending_doc(self, pid: int) -> PendingDoc | None:
        r = self.db.execute("SELECT * FROM pending_docs WHERE id=?", (pid,)).fetchone()
        return self._pending(r) if r else None

    def oldest_pending_doc(self, user_id: int) -> PendingDoc | None:
        r = self.db.execute("SELECT * FROM pending_docs WHERE user_id=? ORDER BY id LIMIT 1", (user_id,)).fetchone()
        return self._pending(r) if r else None

    def pending_doc_by_sha(self, user_id: int, sha: str) -> PendingDoc | None:
        r = self.db.execute("SELECT * FROM pending_docs WHERE user_id=? AND sha256=?", (user_id, sha)).fetchone()
        return self._pending(r) if r else None

    def delete_pending_doc(self, pid: int) -> None:
        self.db.execute("DELETE FROM pending_docs WHERE id=?", (pid,))
        self.db.commit()

    @staticmethod
    def _pending(r: sqlite3.Row) -> PendingDoc:
        return PendingDoc(id=r["id"], user_id=r["user_id"], chat_id=r["chat_id"], title=r["title"],
                          filename=r["filename"], sender_name=r["sender_name"], sent_at=r["sent_at"],
                          text=r["text"], ocr=bool(r["ocr"]), sha256=r["sha256"])

    # ---- pending questions ----
    def add_pending_question(self, user_id: int, question: str) -> int:
        cur = self.db.execute("INSERT INTO pending_questions(user_id, question, created_at) VALUES (?,?,?)",
                              (user_id, question, now_iso()))
        self.db.commit()
        return int(cur.lastrowid)

    def pop_pending_question(self, qid: int) -> tuple[int, str] | None:
        r = self.db.execute("SELECT user_id, question FROM pending_questions WHERE id=?", (qid,)).fetchone()
        if not r:
            return None
        self.db.execute("DELETE FROM pending_questions WHERE id=?", (qid,))
        self.db.commit()
        return r["user_id"], r["question"]

    # ---- user context ----
    def set_context(self, user_id: int, question: str, places: list[str]) -> None:
        self.db.execute(
            "INSERT INTO user_context(user_id, last_question, last_places, updated_at) VALUES (?,?,?,?)"
            " ON CONFLICT(user_id) DO UPDATE SET last_question=excluded.last_question,"
            " last_places=excluded.last_places, updated_at=excluded.updated_at",
            (user_id, question, json.dumps(places, ensure_ascii=False), now_iso()),
        )
        self.db.commit()

    def get_context(self, user_id: int) -> tuple[str, list[str], str] | None:
        r = self.db.execute("SELECT * FROM user_context WHERE user_id=?", (user_id,)).fetchone()
        if not r or not r["last_question"]:
            return None
        return r["last_question"], json.loads(r["last_places"] or "[]"), r["updated_at"]
