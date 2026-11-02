"""Місця лісу: рукавички, частини лісу і відповідність між ними (з «Карти лісу»)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

FOREST = "весь ліс"
PART_CONIFER = "хвойний ліс"
PART_DECIDUOUS = "листяний ліс"
PART_RIVER = "біля річки"
PARTS = [PART_CONIFER, PART_DECIDUOUS, PART_RIVER]

# Рукавички з архіву і запасна відповідність, якщо карту не вдалося прочитати.
DEFAULT_GLOVES: dict[str, list[str]] = {
    "Під ялиною": [PART_CONIFER],
    "У соснах": [PART_CONIFER],
    "Біля мурашника": [PART_CONIFER],
    "Під дубом": [PART_DECIDUOUS],
    "У ліщині": [PART_DECIDUOUS],
    "Під вербою": [PART_DECIDUOUS, PART_RIVER],
    "Біля бобрової греблі": [PART_RIVER],
    "На узліссі": [],
}

# Короткі коди для кнопок Telegram (callback_data обмежена 64 байтами).
CODES = {
    "Під ялиною": "yal", "У соснах": "sos", "Біля мурашника": "mur", "Під дубом": "dub",
    "У ліщині": "lis", "Під вербою": "ver", "Біля бобрової греблі": "gre", "На узліссі": "uzl",
    PART_CONIFER: "hv", PART_DECIDUOUS: "ly", PART_RIVER: "ri", FOREST: "all",
}

# Корені для впізнавання місць у вільному тексті (запасний варіант без LLM).
_GLOVE_PATTERNS = {
    "Під ялиною": r"ялин",
    "У соснах": r"\bу\s+соснах|\bсосн[аиоуі]\w*\s+рукавич|рукавич\w*\s+[«\"]?у\s+соснах",
    "Біля мурашника": r"мурашник",
    "Під дубом": r"\bдуб",
    "У ліщині": r"ліщин",
    "Під вербою": r"\bверб",
    "Біля бобрової греблі": r"гребл|бобров",
    "На узліссі": r"узлісс",
}
_PART_PATTERNS = {
    PART_CONIFER: r"хвойн",
    PART_DECIDUOUS: r"листян",
    PART_RIVER: r"річк|\bріці\b|прибереж",
}


def norm(text: str) -> str:
    return text.lower().replace("’", "'").replace("ʼ", "'").replace("`", "'")


@dataclass
class ForestMap:
    gloves: dict[str, list[str]] = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_GLOVES.items()})

    # ---- карта ----
    @classmethod
    def from_map_text(cls, text: str) -> "ForestMap":
        """Читає легенду карти: «Під ялиною — хвойний ліс.» тощо."""
        fm = cls()
        found: dict[str, list[str]] = {}
        for line in text.splitlines():
            m = re.match(r"^\s*[-–—•]?\s*(.+?)\s+[—–-]\s+(.+?)\.?\s*$", line)
            if not m:
                continue
            name, desc = m.group(1).strip(" «»\""), norm(m.group(2))
            glove = next((g for g in fm.gloves if norm(g) == norm(name)), None)
            if glove is None and not re.match(r"^(Під|У|Біля|На|Над|За|Між|Коло)\s", name):
                continue
            glove = glove or name
            parts: list[str] = []
            if re.search(r"\bні\b", desc):
                parts = []
            else:
                if "хвойн" in desc:
                    parts.append(PART_CONIFER)
                if "листян" in desc:
                    parts.append(PART_DECIDUOUS)
                if "річк" in desc:
                    parts.append(PART_RIVER)
                if not parts:
                    continue
            found[glove] = parts
        if len(found) >= 3:
            fm.gloves = {**fm.gloves, **found}
        return fm

    # ---- місця ----
    @property
    def all_places(self) -> list[str]:
        return list(self.gloves) + PARTS + [FOREST]

    def canonical(self, name: str) -> str | None:
        n = norm(name).strip(" «»\".")
        n = re.sub(r"^(рукавичк[аиуі]|рукавичці)\s+", "", n)
        for p in self.all_places:
            if norm(p) == n:
                return p
        aliases = {"хвойний": PART_CONIFER, "листяний": PART_DECIDUOUS, "річка": PART_RIVER,
                   "біля річки": PART_RIVER, "ліс": FOREST, "весь ліс": FOREST, "увесь ліс": FOREST}
        if n in aliases:
            return aliases[n]
        for p in self.all_places:
            if norm(p) in n:
                return p
        return None

    def code(self, place: str) -> str:
        if place in CODES:
            return CODES[place]
        return "g" + str(list(self.gloves).index(place))

    def by_code(self, code: str) -> str | None:
        for p in self.all_places:
            if self.code(p) == code:
                return p
        return None

    def applicable_scopes(self, places: list[str]) -> set[str]:
        """Для запитаних місць — усі області, поради яких їх стосуються.

        Рукавичка → вона сама + її частини лісу + весь ліс.
        Частина лісу → вона сама + весь ліс.
        """
        scopes = {FOREST}
        for p in places:
            scopes.add(p)
            if p in self.gloves:
                scopes.update(self.gloves[p])
        return scopes

    def detect_places(self, text: str) -> list[str]:
        """Пошук місць у тексті за коренями слів (без LLM)."""
        t = norm(text)
        res = [g for g, pat in _GLOVE_PATTERNS.items() if g in self.gloves and re.search(pat, t)]
        res += [g for g in self.gloves if g not in _GLOVE_PATTERNS and norm(g) in t]
        res += [p for p, pat in _PART_PATTERNS.items() if re.search(pat, t)]
        return res

    def describe(self) -> str:
        lines = []
        for g, parts in self.gloves.items():
            lines.append(f"- рукавичка «{g}»: " + (", ".join(parts) if parts else "не належить до жодної частини лісу"))
        return "\n".join(lines)
