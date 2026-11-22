"""记录最近一次转写结果，供回声处理器匹配（STT 在 handler 之前已替换 Record）。"""

from __future__ import annotations

import time

_TTL = 1800.0
_recent: list[dict] = []


def put_recent(kind: str, text: str) -> None:
    _recent.append({"kind": kind, "text": text, "ts": time.time()})
    if len(_recent) > 20:
        del _recent[: len(_recent) - 20]


def take_recent(message_str: str) -> dict | None:
    """找一条刚转写、且出现在本条消息里的文本，命中即移除。"""
    now = time.time()
    _recent[:] = [r for r in _recent if now - r["ts"] <= _TTL]
    msg = (message_str or "").strip()
    for i, item in enumerate(_recent):
        if item["text"] and item["text"] in msg:
            return _recent.pop(i)
    return None


def peek_latest() -> dict | None:
    """取最近一条（不移除），5 分钟内有效。"""
    now = time.time()
    _recent[:] = [r for r in _recent if now - r["ts"] <= _TTL]
    return _recent[-1] if _recent else None


def clear() -> None:
    _recent.clear()
