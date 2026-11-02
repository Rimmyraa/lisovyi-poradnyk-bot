"""Обробники Telegram (aiogram 3)."""
from __future__ import annotations

import io
import logging
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.enums import ChatAction, ChatType
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from .documents import DocumentService
from .forest_map import FOREST, PARTS, ForestMap, norm
from .qa import QA, scope_label

log = logging.getLogger(__name__)

START = """Привіт! Я «Лісовий порадник» 🌲

Я знаю поради з «Лісового архіву» і з документів, які надсилають звірі.

❓ Питай, наприклад:
• Коли утеплювати рукавичку під ялиною?
• Де шукати гриби біля річки?
• Які внески в рукавичці «У ліщині»?
• Що робити при пожежі?

Назви в питанні рукавичку чи частину лісу, тоді відповідь буде точнішою. До кожної поради я кажу, з якого вона документа.

📄 Маєш нову пораду? Надішли її документом (PDF, Word .docx або .txt), і я її запам'ятаю для всіх.

/places: рукавички й частини лісу
/help: ця довідка"""


def sender_name(msg: Message) -> str:
    u = msg.from_user
    if u is None:
        return "невідомий звір"
    name = u.full_name or "звір"
    return f"{name} (@{u.username})" if u.username else name


def places_keyboard(fmap: ForestMap, prefix: str, item_id: int, with_forest: bool) -> InlineKeyboardMarkup:
    gloves = list(fmap.gloves)
    rows = [[InlineKeyboardButton(text=f"«{g}»", callback_data=f"{prefix}:{item_id}:{fmap.code(g)}")
             for g in gloves[i:i + 2]] for i in range(0, len(gloves), 2)]
    rows.append([InlineKeyboardButton(text=p.capitalize(), callback_data=f"{prefix}:{item_id}:{fmap.code(p)}")
                 for p in PARTS])
    if with_forest:
        rows.append([InlineKeyboardButton(text="🌳 Для всього лісу", callback_data=f"{prefix}:{item_id}:{fmap.code(FOREST)}")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def place_from_text(text: str, fmap: ForestMap) -> list[str]:
    """Розбирає коротку відповідь на питання «для якої рукавички?»."""
    t = norm(text)
    if any(w in t for w in ("весь ліс", "всього лісу", "усього лісу", "для всіх", "увесь ліс")):
        return [FOREST]
    c = fmap.canonical(text)
    if c:
        return [c]
    return fmap.detect_places(text)


def build_router(qa: QA, docs: DocumentService, fmap: ForestMap, tmp_dir: Path, max_file_mb: int = 20) -> Router:
    r = Router()
    r.message.filter(F.chat.type == ChatType.PRIVATE)

    async def send_answer(msg: Message, user_id: int, question: str, forced: list[str] | None = None) -> None:
        await msg.bot.send_chat_action(msg.chat.id, ChatAction.TYPING)
        reply = await qa.answer(user_id, question, forced)
        kb = places_keyboard(fmap, "q", reply.ask_place_qid, False) if reply.ask_place_qid else None
        await msg.answer(reply.text, reply_markup=kb)

    @r.message(CommandStart())
    @r.message(Command("help"))
    async def start(msg: Message) -> None:
        await msg.answer(START)

    @r.message(Command("places"))
    async def places(msg: Message) -> None:
        await msg.answer("Рукавички і частини лісу (з «Карти лісу»):\n" + fmap.describe()
                         + f"\n\nУ пам'яті документів: {qa.storage.count()}.")

    @r.message(F.document)
    async def on_document(msg: Message, bot: Bot) -> None:
        d = msg.document
        if d.file_size and d.file_size > max_file_mb * 1024 * 1024:
            await msg.answer(f"Файл завеликий (більше {max_file_mb} МБ). Надішли, будь ласка, менший документ.")
            return
        await bot.send_chat_action(msg.chat.id, ChatAction.TYPING)
        buf = io.BytesIO()
        await bot.download(d, destination=buf)
        filename = d.file_name or "документ"
        res = await docs.handle_upload(
            user_id=msg.from_user.id, chat_id=msg.chat.id, sender_name=sender_name(msg), filename=filename,
            data=buf.getvalue(), caption=msg.caption or "",
            sent_at=msg.date.isoformat(timespec="seconds"), tmp_dir=tmp_dir,
        )
        if res.status == "saved":
            qa.reindex()
            await msg.answer(f"Дякую! Запам'ятав «{res.title}»: це порада {scope_label(res.document.places)}. "
                             f"Тепер я розповідатиму про неї іншим звірам і казатиму, що її надіслав(-ла) ти.")
        elif res.status == "duplicate":
            await msg.answer(f"Цей документ у мене вже є: «{res.title}» ({scope_label(res.document.places)}).")
        elif res.status == "need_place":
            await msg.answer(f"Я прочитав «{res.title}», але з тексту не зрозуміло, для кого ця порада, "
                             f"а вгадувати я не хочу. Для якої рукавички чи частини лісу вона?",
                             reply_markup=places_keyboard(fmap, "dp", res.pending.id, True))
        elif res.status == "unsupported":
            await msg.answer("Я приймаю документи у форматах PDF, Word (.docx) або текстові файли (.txt). "
                             "Якщо це старий .doc, збережи його як .docx і надішли ще раз.")
        else:
            await msg.answer(f"Не вдалося прочитати текст у «{res.title}». Перевір, будь ласка, що документ не "
                             f"порожній і не пошкоджений, і надішли ще раз.")

    @r.message(F.photo)
    async def on_photo(msg: Message) -> None:
        await msg.answer("Фото я не приймаю 🙂 Надішли пораду документом: PDF, Word (.docx) або .txt.")

    @r.callback_query(F.data.startswith("dp:"))
    async def on_doc_place(cb: CallbackQuery) -> None:
        _, pid, code = cb.data.split(":", 2)
        pending = qa.storage.get_pending_doc(int(pid))
        place = fmap.by_code(code)
        if not pending or pending.user_id != cb.from_user.id or not place:
            await cb.answer("Цей документ уже збережено або запит застарів.", show_alert=False)
            return
        doc = docs.resolve_pending(pending, [place])
        qa.reindex()
        await cb.message.edit_text(f"Дякую! Запам'ятав «{doc.title}»: це порада {scope_label(doc.places)}.")
        await cb.answer()

    @r.callback_query(F.data.startswith("q:"))
    async def on_question_place(cb: CallbackQuery) -> None:
        _, qid, code = cb.data.split(":", 2)
        item = qa.storage.pop_pending_question(int(qid))
        place = fmap.by_code(code)
        if not item or item[0] != cb.from_user.id or not place:
            await cb.answer("Це питання вже застаріло. Спитай ще раз 🙂")
            return
        await cb.answer()
        await cb.message.edit_text(f"{cb.message.text}\n➡️ {place}")
        await send_answer(cb.message, cb.from_user.id, item[1], [place])

    @r.message(F.text)
    async def on_text(msg: Message) -> None:
        text = msg.text.strip()
        pending = qa.storage.oldest_pending_doc(msg.from_user.id)
        if pending and "?" not in text and len(text.split()) <= 6:
            places = place_from_text(text, fmap)
            if places:
                doc = docs.resolve_pending(pending, places)
                qa.reindex()
                await msg.answer(f"Дякую! Запам'ятав «{doc.title}»: це порада {scope_label(doc.places)}.")
                return
            await msg.answer(f"Не впізнав місця. Для кого порада «{pending.title}»? Обери, будь ласка:",
                             reply_markup=places_keyboard(fmap, "dp", pending.id, True))
            return
        await send_answer(msg, msg.from_user.id, text)

    @r.message()
    async def other(msg: Message) -> None:
        await msg.answer("Я розумію текстові питання і документи (PDF, .docx, .txt). /help — довідка.")

    return r
