"""统一的推理线程池：固定 2 个工作线程 + 并发闸门，避免线程随并发增长。"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

# 固定大小：语音推理是 CPU 密集，2 路并行足够，且能压住内存峰值
MAX_WORKERS = 2

_pool: ThreadPoolExecutor | None = None
_sem: asyncio.Semaphore | None = None
_lock = threading.Lock()


def pool() -> ThreadPoolExecutor:
    global _pool
    with _lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(
                max_workers=MAX_WORKERS, thread_name_prefix="asr-enhance"
            )
        return _pool


def _semaphore() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(MAX_WORKERS)
    return _sem


async def run_blocking(func, *args):
    """在固定线程池里跑阻塞任务，并限制同时进行的数量。"""
    async with _semaphore():
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(pool(), func, *args)


def shutdown() -> None:
    global _pool
    with _lock:
        if _pool is not None:
            _pool.shutdown(wait=False)
            _pool = None
