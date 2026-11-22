"""短时间内复用转写结果：同一段语音被多次解析时不重复推理。"""

from __future__ import annotations

import time

_TTL = 300.0
_cache: dict[str, tuple[dict, float]] = {}


def put(key: str, result: dict) -> None:
    if not key:
        return
    _cache[key] = (result, time.time() + _TTL)
    if len(_cache) > 64:
        now = time.time()
        for k in [k for k, (_, exp) in _cache.items() if exp < now]:
            _cache.pop(k, None)


def get(key: str) -> dict | None:
    if not key:
        return None
    item = _cache.get(key)
    if not item:
        return None
    result, exp = item
    if exp < time.time():
        _cache.pop(key, None)
        return None
    return result


def clear() -> None:
    _cache.clear()
