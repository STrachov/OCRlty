from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


def now_iso_ms() -> str:
    """ISO8601 UTC timestamp with milliseconds."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time()%1)*1000):03d}Z"


def get_env_name() -> str:
    return os.getenv("APP_ENV", os.getenv("ENV", "dev"))


def get_log_format() -> str:
    return os.getenv("LOG_FORMAT", "json").lower()  # json | text


def get_log_level() -> str:
    return os.getenv("LOG_LEVEL", os.getenv("LOGLEVEL", "INFO")).upper()


def ensure_basic_configured() -> None:
    """Configure root logging once (idempotent)."""
    root = logging.getLogger()
    if root.handlers:
        return
    log_level = get_log_level()
    log_format = get_log_format()
    logging.basicConfig(
        level=log_level,
        format="%(message)s" if log_format == "json" else "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@dataclass(frozen=True)
class EventLogger:
    service: str
    env: str
    log_format: str
    logger: logging.Logger

    def log_event(
        self,
        level: str,
        event: str,
        request_id: Optional[str] = None,
        msg: str = "",
        **fields: Any,
    ) -> None:
        """One-line structured logging. Keep payloads small; avoid full OCR and raw images."""
        if self.log_format != "json":
            # human-readable
            base = f"{event}"
            if request_id:
                base += f" rid={request_id}"
            if msg:
                base += f" {msg}"
            if fields:
                base += f" {fields}"
            getattr(self.logger, level.lower(), self.logger.info)(base)
            return

        rec: Dict[str, Any] = {
            "ts": now_iso_ms(),
            "level": level.upper(),
            "service": self.service,
            "env": self.env,
            "event": event,
            "msg": msg,
        }
        if request_id:
            rec["request_id"] = request_id
        if fields:
            rec.update(fields)

        line = json.dumps(rec, ensure_ascii=False, default=str)
        getattr(self.logger, level.lower(), self.logger.info)(line)


def get_event_logger(service: str, logger_name: Optional[str] = None) -> EventLogger:
    ensure_basic_configured()
    env = get_env_name()
    log_format = get_log_format()
    logger = logging.getLogger(logger_name or service)
    return EventLogger(service=service, env=env, log_format=log_format, logger=logger)
