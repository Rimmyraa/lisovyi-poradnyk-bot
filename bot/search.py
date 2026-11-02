"""Пошук фрагментів документів: фільтр за місцем + BM25 з простим стемінгом для української."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .forest_map import norm
from .storage import Document

STOP = set("""
і й та а але або чи що як де коли хто це цей ця ці той та ті там тут то так не ні ж же би б бо до від на в у з із зі
за про під над при для по через після перед між його її їх їм їй ми ви вони він вона воно я ти мене тебе нам вам
свій своя свої своє наш наша наші ваш ваша ваші є був була були буде мають має можна треба також ще вже лише тільки
усі всі весь кожен кожна кожного який яка яке які якщо щоб чим чому скільки ось хай нехай
""".split())

_TOKEN = re.compile(r"[a-zа-яіїєґ']+")


def tokens(text: str) -> list[str]:
    out = []
    for w in _TOKEN.findall(norm(text)):
        w = w.strip("'")
        if len(w) < 2 or w in STOP:
            continue
        # Два «стеми»: короткий (повнота) і довший (точність).
        out.append(w[:4])
        if len(w) > 5:
            out.append("~" + w[:6])
    return out


@dataclass
class Chunk:
    doc: Document
    text: str
    idx: int


def split_document(doc: Document, max_chars: int = 1400) -> list[Chunk]:
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[-–•])", doc.text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) > max_chars:
            chunks.append(buf)
            buf = ""
        buf = f"{buf}\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return [Chunk(doc, c, i) for i, c in enumerate(chunks or [doc.text])]


class Index:
    def __init__(self, documents: list[Document]):
        self.chunks: list[Chunk] = []
        for d in documents:
            self.chunks.extend(split_document(d))
        self._tf = [Counter(tokens(c.doc.title + " " + c.doc.title + "\n" + c.text)) for c in self.chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / max(1, len(self._len))
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = max(1, len(self.chunks))
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, scopes: set[str] | None = None, k1: float = 1.4, b: float = 0.75
               ) -> list[tuple[float, Chunk]]:
        q = tokens(query)
        res = []
        for i, c in enumerate(self.chunks):
            if scopes is not None and not (set(c.doc.places) & scopes):
                continue
            tf, ln = self._tf[i], self._len[i]
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self._idf.get(t, 0) * f * (k1 + 1) / (f + k1 * (1 - b + b * ln / self._avg))
            res.append((s, c))
        res.sort(key=lambda x: -x[0])
        return res
