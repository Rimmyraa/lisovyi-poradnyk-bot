"""Відповіді на питання звірів за документами з посиланням на джерела."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from .forest_map import FOREST, PARTS, ForestMap
from .llm import LLM, LLMError, parse_json
from .search import Chunk, Index
from .storage import Document, Storage

log = logging.getLogger(__name__)

NEED_PLACE = "NEED_PLACE"
FOLLOWUP_WINDOW = timedelta(minutes=30)


@dataclass
class Reply:
    text: str
    ask_place_qid: int | None = None   # якщо треба запропонувати кнопки з місцями
    places: list[str] = field(default_factory=list)


def place_label(places: list[str], fmap: ForestMap) -> str:
    out = []
    for p in places:
        if p == FOREST:
            out.append("весь ліс")
        elif p in PARTS:
            out.append(p if p == "біля річки" else p)
        else:
            out.append(f"рукавичка «{p}»")
    return ", ".join(out)


def scope_label(places: list[str]) -> str:
    out = []
    for p in places:
        if p == FOREST:
            out.append("для всього лісу")
        elif p == "біля річки":
            out.append("для тих, хто біля річки")
        elif p in PARTS:
            out.append("для " + p.replace("ий ліс", "ого лісу"))
        else:
            out.append(f"для рукавички «{p}»")
    return ", ".join(out)


def fmt_date(iso: str, tz: str) -> str:
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(ZoneInfo(tz)).strftime("%d.%m.%Y о %H:%M")
    except Exception:
        return iso


def source_line(doc: Document, tz: str) -> str:
    s = f"«{doc.title}» ({scope_label(doc.places)})"
    if doc.source == "telegram":
        s += f" — надіслав(-ла) {doc.sender_name or 'невідомий звір'} {fmt_date(doc.sent_at, tz)}"
    else:
        s += " — з «Лісового архіву»"
    return s


PARSE_PROMPT = """Ти розбираєш питання звіра до бота з порадами про життя в лісі.
Відомі місця:
- рукавички: {gloves}
- частини лісу: "хвойний ліс", "листяний ліс", "біля річки"
- "весь ліс"

Визнач, про які місця питає звір. Рукавичку можуть назвати неточно: «під дубом», «біля греблі», «в ліщині», «у вербі».
Згадка дерева чи предмета ще не означає рукавичку (наприклад, «жолуді з дуба» — це не рукавичка «Під дубом»).
Якщо звір називає місце, якого немає серед відомих, впиши його в unknown_places.
Якщо питання — продовження попереднього («а взимку?», «а там?») і в ньому немає місця, постав followup=true.
У search_query перепиши питання ключовими словами з синонімами (для пошуку в документах).

Відповідь — лише JSON: {{"places": ["точна назва з переліку"], "unknown_places": [], "followup": false, "search_query": "..."}}"""

ANSWER_PROMPT = """Ти — «Лісовий порадник», бот, який відповідає звірам лісу на основі документів, що надіслали інші звірі.

Відповідність рукавичок частинам лісу (з «Карти лісу»):
{map}

Питання стосується: {where}.
Нижче — фрагменти документів, які стосуються цього місця. Кожен має номер [n], область дії і дату.

Правила відповіді:
1. Відповідай українською, коротко, дружньо і по суті питання. Використовуй ЛИШЕ надані фрагменти, нічого не вигадуй.
2. Після КОЖНОЇ поради одразу став номер джерела у квадратних дужках, наприклад: «Утеплюйте в жовтні [2].»
3. Якщо поради суперечать одна одній, перевага за конкретнішою: спершу порада для рукавички, потім для частини лісу, і лише тоді загальна для всього лісу. Давай ЛИШЕ пораду-переможницю, а загальнішу, що їй суперечить, не згадуй. Наприклад, якщо для частини лісу сказано робити щось в одному місяці, а для всього лісу — в іншому, то для рукавички в цій частині лісу відповідь — місяць із поради для частини лісу.
4. Якщо питання про небезпеку чи щось термінове — почни з головної дії.
5. Не переказуй того, що не стосується питання.
6. {no_answer}"""

NO_ANSWER_WITH_PLACE = "Якщо у фрагментах немає відповіді, чесно скажи, що в документах про це нічого немає."
NO_ANSWER_NO_PLACE = (f"Звір не назвав місця, тож тобі дано лише поради для всього лісу. Якщо вони не відповідають "
                      f"на питання (відповідь залежить від рукавички чи частини лісу), відповідай рівно одним словом: {NEED_PLACE}")


class QA:
    def __init__(self, storage: Storage, fmap: ForestMap, llm: LLM, tz: str = "Europe/Kyiv",
                 max_context_chars: int = 9000):
        self.storage, self.fmap, self.llm, self.tz = storage, fmap, llm, tz
        self.max_context_chars = max_context_chars
        self.reindex()

    def reindex(self) -> None:
        self.index = Index(self.storage.all_documents())

    # ---- розбір питання ----
    async def parse(self, question: str, prev: tuple[str, list[str], str] | None) -> dict:
        user = f"Питання: {question}"
        if prev:
            user = f"Попереднє питання: {prev[0]} (місця: {', '.join(prev[1]) or 'не вказано'})\n" + user
        msgs = [{"role": "system", "content": PARSE_PROMPT.format(gloves=", ".join(f"«{g}»" for g in self.fmap.gloves))},
                {"role": "user", "content": user}]
        try:
            data = parse_json(await self.llm.chat(msgs, fast=True, json_mode=True, max_tokens=250))
            places = []
            for p in data.get("places") or []:
                c = self.fmap.canonical(str(p))
                if c and c not in places:
                    places.append(c)
            return {"places": places, "unknown": [str(u) for u in data.get("unknown_places") or [] if str(u).strip()],
                    "followup": bool(data.get("followup")), "query": str(data.get("search_query") or "")}
        except LLMError as e:
            log.warning("Розбір питання без LLM: %s", e)
            return {"places": self.fmap.detect_places(question), "unknown": [], "followup": False, "query": ""}

    # ---- відповідь ----
    async def answer(self, user_id: int, question: str, forced_places: list[str] | None = None) -> Reply:
        prev = self.storage.get_context(user_id)
        if prev:
            try:
                if datetime.now(timezone.utc) - datetime.fromisoformat(prev[2]) > FOLLOWUP_WINDOW:
                    prev = None
            except ValueError:
                prev = None
        if forced_places is not None:
            parsed = {"places": forced_places, "unknown": [], "followup": False, "query": ""}
        else:
            parsed = await self.parse(question, prev)
        places = [p for p in parsed["places"] if p != FOREST] if len(parsed["places"]) > 1 else parsed["places"]
        if not places and parsed["followup"] and prev and prev[1]:
            places = prev[1]
            question = f"{prev[0]} — {question}"
        if not places and parsed["unknown"]:
            known = ", ".join(f"«{g}»" for g in self.fmap.gloves)
            return Reply(f"Я не знаю місця «{parsed['unknown'][0]}». Можу підказати для рукавичок: {known}, "
                         f"а також для хвойного лісу, листяного лісу, тих, хто біля річки, чи для всього лісу.")
        query = f"{question} {parsed['query']}".strip()

        if places and places != [FOREST]:
            reply = await self._answer_for(question, query, places)
            self.storage.set_context(user_id, question, places)
            return reply

        # Місце не назване: відповідаємо за порадами для всього лісу, а якщо їх замало — перепитуємо.
        forest_hits = [(s, c) for s, c in self.index.search(query, {FOREST}) if s > 0]
        if forest_hits:
            reply = await self._answer_for(question, query, [FOREST], no_place=True, hits=forest_hits)
            if reply is not None:
                specific = [c for s, c in self.index.search(query) if s > 0 and FOREST not in c.doc.places][:5]
                if specific and forced_places is None:
                    reply.text += ("\n\nЦе поради для всього лісу. Якщо скажеш, про яку рукавичку чи частину лісу "
                                   "питаєш, підкажу точніше.")
                self.storage.set_context(user_id, question, [FOREST])
                return reply
        qid = self.storage.add_pending_question(user_id, question)
        return Reply("Відповідь залежить від місця. Про яку рукавичку чи частину лісу питаєш?", ask_place_qid=qid)

    def _select(self, hits: list[tuple[float, Chunk]]) -> list[Chunk]:
        chosen, total = [], 0
        top = hits[0][0] if hits else 0
        for s, c in hits:
            if s <= 0 or s < top * 0.12:
                break
            if total + len(c.text) > self.max_context_chars and chosen:
                continue
            chosen.append(c)
            total += len(c.text)
            if len(chosen) >= 14:
                break
        return chosen

    async def _answer_for(self, question: str, query: str, places: list[str], no_place: bool = False,
                          hits: list[tuple[float, Chunk]] | None = None) -> Reply | None:
        scopes = self.fmap.applicable_scopes(places)
        hits = hits if hits is not None else self.index.search(query, scopes)
        chunks = self._select(hits)
        where = place_label(places, self.fmap)
        if not chunks:
            if no_place:
                return None
            return Reply(f"У документах поки немає порад про це для: {where}.", places=places)

        docs: list[Document] = []
        for c in chunks:
            if c.doc not in docs:
                docs.append(c.doc)
        ctx = []
        for i, d in enumerate(docs, 1):
            body = "\n".join(c.text for c in chunks if c.doc is d)
            origin = (f"надіслав {d.sender_name}, {fmt_date(d.sent_at, self.tz)}" if d.source == "telegram"
                      else f"Лісовий архів, {fmt_date(d.sent_at, self.tz)[:10]}")
            ctx.append(f"[{i}] «{d.title}» — {scope_label(d.places)}; {origin}\n{body}")
        msgs = [
            {"role": "system", "content": ANSWER_PROMPT.format(
                map=self.fmap.describe(), where=where,
                no_answer=NO_ANSWER_NO_PLACE if no_place else NO_ANSWER_WITH_PLACE)},
            {"role": "user", "content": "Фрагменти:\n\n" + "\n\n".join(ctx) + f"\n\nПитання звіра: {question}"},
        ]
        try:
            text = await self.llm.chat(msgs, max_tokens=900)
        except LLMError as e:
            log.error("LLM не відповів: %s", e)
            return Reply("Вибач, зараз не можу дістатися до своєї мудрості (сервіс перевантажений). "
                         "Спробуй, будь ласка, за хвилину.", places=places)
        if no_place and NEED_PLACE in text:
            return None
        return Reply(self._render(text, docs), places=places)

    def _render(self, text: str, docs: list[Document]) -> str:
        """Перенумеровує посилання [n] за порядком появи і додає список джерел."""
        order: list[int] = []

        def repl(m: re.Match) -> str:
            nums = [int(x) for x in re.findall(r"\d+", m.group(0))]
            out = []
            for n in nums:
                if 1 <= n <= len(docs):
                    if n not in order:
                        order.append(n)
                    out.append(str(order.index(n) + 1))
            return f"[{', '.join(out)}]" if out else ""

        body = re.sub(r"\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]", repl, text).strip()
        if not order:
            return body
        lines = [f"[{i}] {source_line(docs[n - 1], self.tz)}" for i, n in enumerate(order, 1)]
        return body + "\n\n📄 Джерела:\n" + "\n".join(lines)
