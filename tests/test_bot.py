"""Тести логіки бота без мережі: справжній архів + підставна LLM."""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from pathlib import Path

import pytest

from bot.classify import classify_rules
from bot.documents import DocumentService
from bot.forest_map import FOREST, PART_CONIFER, PART_DECIDUOUS, PART_RIVER, ForestMap
from bot.qa import QA, NEED_PLACE
from bot.storage import Storage

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "data" / "archive"
TESSDATA = os.getenv("TESSDATA_PREFIX", "")


class FakeLLM:
    """Імітує Groq: розбирає питання коренями слів, відповідає першим фрагментом."""

    def __init__(self, fmap: ForestMap, classify_answer: dict | None = None, need_place: bool = False):
        self.fmap, self.classify_answer, self.need_place = fmap, classify_answer, need_place
        self.calls: list[str] = []

    async def chat(self, messages, *, fast=False, json_mode=False, max_tokens=900):
        system, user = messages[0]["content"], messages[-1]["content"]
        if "розбираєш питання" in system:
            self.calls.append("parse")
            q = user.split("Питання:")[-1]
            return json.dumps({"places": self.fmap.detect_places(q), "unknown_places": [],
                               "followup": q.strip().startswith("А "), "search_query": ""})
        if "Визнач, для кого" in system:
            self.calls.append("classify")
            return json.dumps(self.classify_answer or {"scope": "unknown", "places": [], "evidence": ""})
        self.calls.append("answer")
        self.last_context = user
        if self.need_place and "для всього лісу" in system and "не назвав місця" in system:
            return NEED_PLACE
        return "Ось порада [1]. А ще ось [2]."


@pytest.fixture(scope="session")
def archive_db(tmp_path_factory):
    """Один раз завантажує весь архів (з OCR) у базу і далі копіює її для кожного тесту."""
    path = tmp_path_factory.mktemp("db") / "archive.db"
    st = Storage(path)
    fm = ForestMap()
    asyncio.run(DocumentService(st, fm, None, TESSDATA).ingest_archive(ARCHIVE))
    st.db.close()
    return path


@pytest.fixture
def env(archive_db, tmp_path):
    db = tmp_path / "bot.db"
    shutil.copy(archive_db, db)
    st = Storage(db)
    fm = ForestMap()
    svc = DocumentService(st, fm, None, TESSDATA)
    svc.load_map()
    return st, fm, svc


def run(coro):
    return asyncio.run(coro)


# ---------- карта і області ----------

def test_map_read_from_document(env):
    st, fm, _ = env
    assert fm.gloves["Під вербою"] == [PART_DECIDUOUS, PART_RIVER]
    assert fm.gloves["На узліссі"] == []
    assert fm.gloves["Біля мурашника"] == [PART_CONIFER]


def test_all_archive_documents_classified(env):
    st, _, _ = env
    docs = st.all_documents()
    assert len(docs) == 78
    by_title = {d.title: d.places for d in docs}
    assert by_title["Що робити при пожежі"] == [FOREST]
    assert by_title["Броди і коли бобри відкривають греблю"] == [PART_RIVER]
    assert by_title["Як не заблукати серед однакових сосен"] == [PART_CONIFER]
    assert by_title["Членські внески рукавички Під ялиною"] == ["Під ялиною"]  # скан, OCR
    assert sum(d.ocr for d in docs) == 19


def test_scans_are_readable(env):
    st, _, _ = env
    doc = next(d for d in st.all_documents() if d.title == "Членські внески рукавички Під ялиною")
    assert "3 шишки" in doc.text and "Білочці" in doc.text


def test_scopes_follow_map(env):
    _, fm, _ = env
    assert fm.applicable_scopes(["Під вербою"]) == {FOREST, "Під вербою", PART_DECIDUOUS, PART_RIVER}
    assert fm.applicable_scopes(["На узліссі"]) == {FOREST, "На узліссі"}


# ---------- відповіді ----------

def test_answer_uses_specific_and_cites_sources(env):
    st, fm, _ = env
    llm = FakeLLM(fm)
    qa = QA(st, fm, llm)
    reply = run(qa.answer(1, "Коли утеплювати рукавичку під ялиною?"))
    ctx = llm.last_context
    assert "Як утеплити рукавичку в хвойному лісі" in ctx and "Календар сезонів" in ctx
    assert "Як утеплити рукавичку в листяному лісі" not in ctx  # ялина — у хвойному лісі
    assert "Як утеплити рукавичку біля річки" not in ctx
    assert "📄 Джерела:" in reply.text
    assert "з «Лісового архіву»" in reply.text


def test_edge_glove_gets_no_part_advice(env):
    st, fm, _ = env
    llm = FakeLLM(fm)
    qa = QA(st, fm, llm)
    run(qa.answer(1, "Де шукати гриби на узліссі?"))
    ctx = llm.last_context
    assert "Грибний путівник" not in ctx           # путівники для частин лісу не стосуються узлісся
    assert "Їстівне й отруйне" in ctx              # а загальні поради — стосуються


def test_question_without_place_answers_from_forest_docs(env):
    st, fm, _ = env
    llm = FakeLLM(fm)
    qa = QA(st, fm, llm)
    reply = run(qa.answer(1, "Що робити при пожежі?"))
    assert reply.ask_place_qid is None
    assert "Що робити при пожежі" in llm.last_context


def test_question_without_place_asks_for_place(env):
    st, fm, _ = env
    llm = FakeLLM(fm, need_place=True)
    qa = QA(st, fm, llm)
    reply = run(qa.answer(1, "Які членські внески і кому їх здавати?"))
    assert reply.ask_place_qid is not None
    uid, q = st.pop_pending_question(reply.ask_place_qid)
    reply2 = run(qa.answer(uid, q, ["У ліщині"]))
    assert "Членські внески рукавички У ліщині" in llm.last_context
    assert "[1]" in reply2.text


def test_followup_keeps_place(env):
    st, fm, _ = env
    llm = FakeLLM(fm)
    qa = QA(st, fm, llm)
    run(qa.answer(7, "Які небезпеки біля рукавички під дубом?"))
    run(qa.answer(7, "А хто там чергує?"))
    assert "Чергування в рукавичці Під дубом" in llm.last_context


def test_render_renumbers_and_drops_bad_refs(env):
    st, fm, _ = env
    qa = QA(st, fm, FakeLLM(fm))
    docs = st.all_documents()[:3]
    out = qa._render("Перше [3]. Друге [1, 3]. Дивне [9].", docs)
    assert out.startswith("Перше [1]. Друге [2, 1]. Дивне .")
    assert out.count("\n[") == 2


# ---------- документи з Telegram ----------

def test_upload_with_clear_place_is_saved_with_sender(env, tmp_path):
    st, fm, svc = env
    svc.llm = FakeLLM(fm)
    text = "Порада для мешканців рукавички «Під дубом».\n\nНова яма з'явилася за 10 кроків на схід."
    res = run(svc.handle_upload(user_id=5, chat_id=5, sender_name="Їжак (@yizhak)", filename="порада.txt",
                                data=text.encode(), sent_at="2026-11-02T20:47:00+00:00", tmp_dir=tmp_path))
    assert res.status == "saved" and res.document.places == ["Під дубом"]
    qa = QA(st, fm, FakeLLM(fm))
    reply = run(qa.answer(1, "Де ями біля рукавички під дубом?"))
    assert "Їжак (@yizhak)" in reply.text or "Їжак (@yizhak)" in qa.llm.last_context
    assert "02.11.2026 о 22:47" in qa.llm.last_context  # час за Києвом


def test_upload_with_unclear_place_asks_sender(env, tmp_path):
    st, fm, svc = env
    svc.llm = FakeLLM(fm)
    text = "Порада від Сови\n\nСвіжі гриби краще сушити на сонці, а не біля вогню."
    res = run(svc.handle_upload(user_id=5, chat_id=5, sender_name="Сова", filename="doc.txt",
                                data=text.encode(), tmp_dir=tmp_path))
    assert res.status == "need_place"
    assert st.count("telegram") == 0
    doc, replaced = svc.resolve_pending(res.pending, ["хвойний ліс"])
    assert replaced == []
    assert doc.places == ["хвойний ліс"] and doc.sender_name == "Сова"
    assert st.oldest_pending_doc(5) is None


def test_upload_caption_sets_place(env, tmp_path):
    st, fm, svc = env
    svc.llm = FakeLLM(fm)
    res = run(svc.handle_upload(user_id=5, chat_id=5, sender_name="Сова", filename="x.txt",
                                data="Сушіть гриби на сонці, не біля вогню.".encode(),
                                caption="це для рукавички У ліщині", tmp_dir=tmp_path))
    assert res.status == "saved" and res.document.places == ["У ліщині"]


def test_duplicate_and_unsupported(env, tmp_path):
    st, fm, svc = env
    data = (ARCHIVE / "Що робити при пожежі.txt").read_bytes()
    res = run(svc.handle_upload(user_id=1, chat_id=1, sender_name="x", filename="a.txt", data=data,
                                tmp_dir=tmp_path))
    assert res.status == "duplicate"
    res = run(svc.handle_upload(user_id=1, chat_id=1, sender_name="x", filename="a.jpg", data=b"123",
                                tmp_dir=tmp_path))
    assert res.status == "unsupported"


def test_classify_rules_cases(env):
    _, fm, _ = env
    assert classify_rules("Порада", "Порада\n\nЦя пам'ятка для всіх мешканців лісу.", fm) == [FOREST]
    assert classify_rules("Порада", "Порада\n\nДля тих, хто живе біля річки: не купайтеся.", fm) == [PART_RIVER]
    assert classify_rules("Корисне", "Корисне\n\nСушіть гриби на сонці.", fm) is None


# ---------- версії документів ----------

def test_new_version_replaces_old_for_same_place(env, tmp_path):
    st, fm, svc = env
    svc.llm = FakeLLM(fm)
    new = ("Членські внески рукавички «Під ялиною»\n\nВнесок тепер 5 шишок з дорослого звіра на місяць. "
           "Здавати Дятлові.")
    res = run(svc.handle_upload(user_id=9, chat_id=9, sender_name="Мишка", data=new.encode(),
                                filename="Членські внески рукавички Під ялиною.txt", tmp_dir=tmp_path))
    assert res.status == "saved" and len(res.replaced) == 1 and res.replaced[0].source == "archive"
    assert st.count() == 78  # стару версію замінено, а не додано ще одну
    llm = FakeLLM(fm)
    qa = QA(st, fm, llm)
    run(qa.answer(1, "Які внески в рукавичці під ялиною?"))
    assert "5 шишок" in llm.last_context and "3 шишки" not in llm.last_context

    # повторне надсилання старої версії не повертає її
    old = (ARCHIVE / "Членські внески рукавички Під ялиною.pdf").read_bytes()
    res2 = run(svc.handle_upload(user_id=9, chat_id=9, sender_name="x", data=old,
                                 filename="Членські внески рукавички Під ялиною.pdf", tmp_dir=tmp_path))
    assert res2.status == "outdated" and "5 шишок" in res2.document.text


def test_same_title_other_place_is_not_replaced(env, tmp_path):
    st, fm, svc = env
    svc.llm = FakeLLM(fm)
    text = "Небезпеки хвойного лісу\n\nПорада для мешканців рукавички «У соснах»: обходьте нову яму."
    res = run(svc.handle_upload(user_id=9, chat_id=9, sender_name="Зайчик", data=text.encode(),
                                filename="x.txt", caption="для рукавички У соснах", tmp_dir=tmp_path))
    assert res.status == "saved" and res.replaced == []
    assert st.count() == 79


def test_title_key_ignores_copy_suffixes():
    from bot.storage import title_key
    assert title_key("Правила толоки (1)") == title_key("правила толоки.txt") == title_key("Правила толоки v2")
