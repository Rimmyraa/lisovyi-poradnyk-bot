"""Визначення, для кого порада в документі: весь ліс, частина лісу чи конкретна рукавичка."""
from __future__ import annotations

import logging
import re

from .forest_map import FOREST, PART_CONIFER, PART_DECIDUOUS, PART_RIVER, ForestMap, norm
from .llm import LLM, LLMError, parse_json

log = logging.getLogger(__name__)

_FOREST_PHRASES = re.compile(
    r"(всіх|усіх|кожного)\s+(мешканц\w*|звір\w*)\s+(нашого\s+)?лісу|всього лісу|усього лісу|"
    r"незалежно від того, де|для всього лісу|всім мешканцям лісу|усім мешканцям лісу"
)
_PARTS = [(PART_CONIFER, r"хвойн"), (PART_DECIDUOUS, r"листян"), (PART_RIVER, r"річк|\bріці\b|прибереж")]


def _intro(text: str, title: str) -> str:
    body = text
    lines = [line for line in text.splitlines() if line.strip()]
    if lines:
        first_line = re.sub(r"[«»\"'’.\s]+", " ", norm(lines[0])).strip()
        t = re.sub(r"[«»\"'’.\s]+", " ", norm(title)).strip()
        if first_line == t or (len(first_line) < 120 and not re.search(r"[.!?]$", lines[0].strip())
                               and len(lines) > 1):
            body = "\n".join(lines[1:])  # перший рядок — заголовок
    # перший абзац / ~2 речення
    first = re.split(r"\n\s*\n", body.strip(), maxsplit=1)[0]
    sentences = re.split(r"(?<=[.!?])\s+", first)
    return " ".join(sentences[:2])[:400]


def _gloves_named(text: str, fmap: ForestMap) -> list[str]:
    t = norm(text)
    found = []
    for g in fmap.gloves:
        if re.search(r"(?<![а-яіїєґ])" + re.escape(norm(g)) + r"(?![а-яіїєґ])", t):
            found.append(g)
    return found


def classify_rules(title: str, text: str, fmap: ForestMap) -> list[str] | None:
    """Надійні ознаки без LLM. Повертає None, якщо однозначної ознаки немає."""
    t_title = norm(title)
    gloves = _gloves_named(title, fmap)
    if gloves:
        return gloves
    parts = [p for p, pat in _PARTS if re.search(pat, t_title)]
    if parts:
        return parts
    intro = norm(_intro(text, title))
    quoted = [g for g in fmap.gloves if f"«{norm(g)}»" in intro or f'"{norm(g)}"' in intro]
    if quoted and re.search(r"рукавич", intro):
        return quoted
    if _FOREST_PHRASES.search(intro):
        return [FOREST]
    parts = [p for p, pat in _PARTS if re.search(pat, intro)]
    if parts and re.search(r"(для|мешканц|живе|живуть|стоїть|лежить)", intro):
        return parts
    return None


CLASSIFY_PROMPT = """Ти допомагаєш боту лісового архіву. Визнач, для кого написана порада в документі.
Можливі області:
- "весь ліс" — порада для всіх мешканців лісу;
- частина лісу: "хвойний ліс", "листяний ліс", "біля річки";
- конкретна рукавичка (громада звірів): {gloves}.

Відповідність рукавичок частинам лісу:
{map}

Правила:
- Визначай ТІЛЬКИ за тим, що прямо написано в документі або в підписі відправника. Не вгадуй.
- Якщо документ прямо адресований кільком областям — перелічи всі.
- Якщо з тексту незрозуміло, для кого порада, поверни scope "unknown" і порожній список.
- У полі evidence наведи коротку цитату з тексту, на якій ґрунтується висновок.

Відповідь — лише JSON: {{"scope": "forest|part|glove|unknown", "places": ["..."], "evidence": "..."}}"""


async def classify_llm(title: str, text: str, caption: str, fmap: ForestMap, llm: LLM) -> list[str] | None:
    user = f"Назва файлу: {title}\n"
    if caption:
        user += f"Підпис відправника: {caption}\n"
    user += f"Текст документа:\n{text[:5000]}"
    msgs = [
        {"role": "system", "content": CLASSIFY_PROMPT.format(
            gloves=", ".join(f"«{g}»" for g in fmap.gloves), map=fmap.describe())},
        {"role": "user", "content": user},
    ]
    try:
        data = parse_json(await llm.chat(msgs, json_mode=True, max_tokens=300))
    except LLMError as e:
        log.warning("Класифікація LLM не вдалася: %s", e)
        return None
    if data.get("scope") == "unknown" or not str(data.get("evidence", "")).strip():
        return None
    places = []
    for p in data.get("places") or []:
        c = fmap.canonical(str(p))
        if c and c not in places:
            places.append(c)
    if FOREST in places and len(places) > 1:
        places.remove(FOREST)
    return places or None


async def classify_document(title: str, text: str, caption: str, fmap: ForestMap, llm: LLM | None
                            ) -> list[str] | None:
    """Порядок: підпис відправника → надійні правила → LLM. None = треба перепитати."""
    if caption.strip():
        # Підпис відправника — пряма вказівка, але приймаємо тільки повні назви місць.
        cap = _gloves_named(caption, fmap) + [p for p, pat in _PARTS if re.search(pat, norm(caption))]
        if not cap and (_FOREST_PHRASES.search(norm(caption)) or re.search(r"(весь|увесь|для всіх)", norm(caption))):
            cap = [FOREST]
        if cap:
            return cap
    places = classify_rules(title, text, fmap)
    if places:
        return places
    if llm is None:
        return None
    return await classify_llm(title, text, caption, fmap, llm)
