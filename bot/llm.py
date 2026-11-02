"""Клієнт Groq (OpenAI-сумісний API) з переходом на запасні моделі при лімітах."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Protocol

log = logging.getLogger(__name__)

API = "https://api.groq.com/openai/v1"


class LLMError(Exception):
    pass


class LLM(Protocol):
    async def chat(self, messages: list[dict], *, fast: bool = False, json_mode: bool = False,
                   max_tokens: int = 900) -> str: ...


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def parse_json(text: str) -> dict[str, Any]:
    text = strip_think(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    raise LLMError(f"Не JSON: {text[:200]}")


class GroqLLM:
    def __init__(self, api_key: str, models: list[str], fast_models: list[str]):
        self.api_key = api_key
        self.models = models
        self.fast_models = fast_models
        self._session = None
        self._unavailable: set[str] = set()

    async def _sess(self):
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=aiohttp.ClientTimeout(total=90),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def check_models(self) -> None:
        """Прибирає зі списків моделі, яких немає у Groq (щоб не витрачати запити)."""
        try:
            s = await self._sess()
            async with s.get(f"{API}/models") as r:
                if r.status != 200:
                    log.warning("Groq /models: %s %s", r.status, await r.text())
                    return
                ids = {m["id"] for m in (await r.json()).get("data", [])}
        except Exception as e:  # мережа тощо — не критично
            log.warning("Не вдалося отримати список моделей Groq: %s", e)
            return
        for lst in (self.models, self.fast_models):
            keep = [m for m in lst if m in ids]
            if keep:
                lst[:] = keep
        log.info("Моделі Groq: основні %s, швидкі %s", self.models, self.fast_models)

    async def chat(self, messages: list[dict], *, fast: bool = False, json_mode: bool = False,
                   max_tokens: int = 900) -> str:
        chain = [m for m in (self.fast_models if fast else self.models) if m not in self._unavailable]
        chain = chain or (self.fast_models if fast else self.models)
        last = ""
        s = await self._sess()
        for model in chain:
            body: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0.1,
                                    "max_tokens": max_tokens}
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            for attempt in range(2):
                try:
                    async with s.post(f"{API}/chat/completions", json=body) as r:
                        txt = await r.text()
                        if r.status == 200:
                            content = json.loads(txt)["choices"][0]["message"]["content"] or ""
                            return strip_think(content)
                        last = f"{model}: {r.status} {txt[:300]}"
                        log.warning("Groq помилка %s", last)
                        if r.status == 429:
                            wait = float(r.headers.get("retry-after", "0") or 0)
                            if attempt == 0 and 0 < wait <= 8:
                                await asyncio.sleep(wait)
                                continue
                            break  # наступна модель
                        if r.status in (400, 404) and ("model" in txt and ("not" in txt or "decommission" in txt)):
                            self._unavailable.add(model)
                            break
                        if r.status == 400 and json_mode and "json" in txt.lower():
                            body.pop("response_format", None)  # модель не підтримує json_mode
                            continue
                        if r.status >= 500 and attempt == 0:
                            await asyncio.sleep(1.5)
                            continue
                        break
                except (asyncio.TimeoutError, OSError) as e:
                    last = f"{model}: {e!r}"
                    log.warning("Groq мережа: %s", last)
                    if attempt == 0:
                        await asyncio.sleep(1.5)
                        continue
                    break
        raise LLMError(last or "немає моделей")
