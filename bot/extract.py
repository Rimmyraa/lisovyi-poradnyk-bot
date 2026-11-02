"""Витяг тексту з PDF, DOCX і TXT. Для сканованих PDF — OCR (tesseract, українська)."""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".docx", ".txt"}


class UnsupportedFormat(Exception):
    pass


def _decode_text(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1251"):
        try:
            text = data.decode(enc)
            if enc == "utf-16" and not data[:2] in (b"\xff\xfe", b"\xfe\xff"):
                continue
            return text
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _docx_text(path: Path) -> str:
    import docx  # python-docx

    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs]
    for table in d.tables:
        for row in table.rows:
            parts.append(" | ".join(c.text for c in row.cells))
    return "\n".join(parts)


def _pdf_pages(path: Path) -> int:
    out = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True).stdout
    m = re.search(r"Pages:\s+(\d+)", out)
    return int(m.group(1)) if m else 1


def _pdf_text(path: Path) -> str:
    return subprocess.run(["pdftotext", "-enc", "UTF-8", str(path), "-"],
                          capture_output=True, text=True).stdout


_LETTERS = re.compile(r"[A-Za-zА-Яа-яЇїІіЄєҐґ'’]{2,}")


def _good_ocr_line(line: str) -> bool:
    """Відкидає «сміття», яке OCR знаходить у фоні (текстура берести)."""
    words = _LETTERS.findall(line)
    letters = sum(len(w) for w in words)
    compact = re.sub(r"\s", "", line)
    return bool(words) and letters >= 4 and letters / max(1, len(compact)) > 0.6


def _clean_ocr(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if not _good_ocr_line(line):
            continue
        line = re.sub(r"^[\W\d_]{1,6}\s+(?=[А-ЯЇІЄҐ])", "", line)  # хвости на початку рядка
        line = re.sub(r"\s+[|\-–—\"'.,]{1,3}(\s+\S{1,3})*\s*$", "", line)  # і в кінці
        lines.append(line.strip())
    return "\n".join(lines)


def ocr_pdf(path: Path, tessdata: str = "", lang: str = "ukr+eng") -> str:
    from PIL import Image  # pillow

    env = dict(os.environ)
    if tessdata:
        env["TESSDATA_PREFIX"] = tessdata
    texts = []
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", "300", "-png", str(path), f"{tmp}/p"], check=True)
        for img in sorted(Path(tmp).glob("p*.png")):
            # Текст у документах темний, фон світлий/бежевий — лишаємо тільки темне.
            Image.open(img).convert("L").point(lambda v: 0 if v < 110 else 255).save(img)
            out = subprocess.run(["tesseract", str(img), "-", "-l", lang, "--psm", "4"],
                                 capture_output=True, text=True, env=env).stdout
            texts.append(_clean_ocr(out))
    return "\n\n".join(texts)


def extract_text(path: Path, tessdata: str = "") -> tuple[str, bool]:
    """Повертає (текст, чи_був_OCR)."""
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise UnsupportedFormat(ext)
    if ext == ".txt":
        return _decode_text(path.read_bytes()), False
    if ext == ".docx":
        return _docx_text(path), False
    text = _pdf_text(path)
    pages = _pdf_pages(path)
    if len(re.sub(r"\s", "", text)) < 80 * pages:
        log.info("PDF без текстового шару, запускаю OCR: %s", path.name)
        ocr = ocr_pdf(path, tessdata)
        if len(ocr) > len(text):
            return ocr, True
    return text, False


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


_GENERIC_NAME = re.compile(r"^(doc|document|file|scan|img|image|new|untitled|документ|файл|скан|без назви)?[\W\d_]*$", re.I)


def make_title(filename: str, text: str) -> str:
    stem = Path(filename).stem.strip()
    if stem and not _GENERIC_NAME.match(stem) and re.search(r"[А-Яа-яЇїІіЄєҐґ]", stem):
        return stem
    for line in text.splitlines():
        line = line.strip(" -–—#*")
        if 3 <= len(line) <= 120 and re.search(r"[А-Яа-яЇїІіЄєҐґA-Za-z]{3}", line):
            return line
    return stem or "Без назви"
