# lib/pipelines/tilt_client.py
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx

MOCK = os.getenv("MOCK_VLLM", "0") == "1"


def _normalize_base_url(url: str) -> str:
    url = url.strip().rstrip("/")
    # допускаем как ...:8001 так и ...:8001/v1
    return url if url.endswith("/v1") else (url + "/v1")


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """Достаём JSON даже если модель вернула его внутри текста/```json```."""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    # ```json ... ```
    import re as _re
    blocks = _re.findall(r"```json(.*?)```", text, flags=_re.DOTALL | _re.IGNORECASE)
    for b in blocks:
        try:
            return json.loads(b.strip())
        except Exception:
            pass
    # первая сбалансированная {...}
    stack, start = [], None
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            stack.append(i)
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    chunk = text[start : i + 1]
                    try:
                        return json.loads(chunk)
                    except Exception:
                        start = None
    # примитивный фолбэк: заменить одинарные кавычки
    try:
        return json.loads(text.replace("'", '"'))
    except Exception:
        return {"raw": text}


class ArcticTiltClient:
    def __init__(
        self,
        base_url: str,
        model: str = "Snowflake/snowflake-arctic-tilt-v1.3",
        timeout: float = 10.0,
        api_key: Optional[str] = None,
        max_retries: int = 1,
        retry_backoff_s: float = 0.5,
    ):
        self.base_url = _normalize_base_url(base_url)
        self.model = model
        self.timeout = timeout
        self.api_key = api_key or os.getenv("VLLM_API_KEY", "")
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self._cli = httpx.Client(timeout=self.timeout)

        # ленивый импорт препроцессора — только в «боевом» режиме
        self._pp = None
        self._pp_err: Optional[Exception] = None
        if not MOCK:
            try:
                # корректный путь для v0.8.3 (как в их examples/tilt_example.py)
                from vllm.multimodal.tilt_processor import TiltPreprocessor  # type: ignore
                self._pp = TiltPreprocessor()
            except Exception as e:
                self._pp_err = e  # не падаем здесь; упадём при первом infer, чтобы health работал

    # ---------------- internal helpers ----------------

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self._cli.post(url, headers=self._headers(), json=payload)
                r.raise_for_status()
                return r.json()
            except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.TransportError) as e:
                last_exc = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_s * (2**attempt))
                else:
                    raise
        # не должно сюда дойти
        if last_exc:
            raise last_exc
        raise RuntimeError("unknown error in _post_chat")

    def _build_messages_mock(self, doc_bytes: bytes) -> List[Dict[str, str]]:
        # минимальные сообщения для мок-сервера
        return [{"role": "user", "content": f"len={len(doc_bytes)} bytes"}]

    def _build_messages_real(self, doc_bytes: bytes) -> List[Dict[str, str]]:
        if self._pp_err:
            raise RuntimeError(f"TiltPreprocessor import/init failed: {self._pp_err!r}")
        if self._pp is None:
            raise RuntimeError("TiltPreprocessor is not initialized")
        # наиболее частая сигнатура в v0.8.3
        if hasattr(self._pp, "build_messages_from_bytes"):
            return self._pp.build_messages_from_bytes(doc_bytes)  # type: ignore
        # альтернативный путь (на всякий случай)
        if hasattr(self._pp, "preprocess") and hasattr(self._pp, "build_messages"):
            doc = self._pp.preprocess(doc_bytes)  # type: ignore
            return self._pp.build_messages(doc)   # type: ignore
        raise AttributeError("TiltPreprocessor API mismatch")

    def _parse_response(self, content: str) -> Dict[str, Any]:
        # если у препроцессора есть свой парсер — предпочитаем его
        if self._pp and hasattr(self._pp, "parse_response"):
            try:
                return self._pp.parse_response(content)  # type: ignore
            except Exception:
                pass
        return _extract_json_from_text(content)

    # ---------------- public API ----------------

    def infer(self, doc_bytes: bytes) -> Dict[str, Any]:
        # 1) messages (mock vs real)
        messages = (
            self._build_messages_mock(doc_bytes)
            if MOCK
            else self._build_messages_real(doc_bytes)
        )

        # 2) запрос в (mock)vLLM
        payload = {"model": self.model, "messages": messages, "temperature": 0.0}
        resp = self._post_chat(payload)

        # 3) извлечь текст
        try:
            content = resp["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Unexpected /chat/completions response: {e}; got: {resp}")

        # 4) в словарь полей
        return self._parse_response(content)

    def close(self) -> None:
        try:
            self._cli.close()
        except Exception:
            pass
