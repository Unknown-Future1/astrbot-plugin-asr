"""音频解析：任意来源 → 16k 单声道 PCM。"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np


def find_ffmpeg(configured: str = "") -> str:
    if configured:
        return configured
    found = shutil.which("ffmpeg")
    if found:
        return found
    # Windows 常见位置兜底
    for candidate in (
        r"H:\Python\tools\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ):
        if Path(candidate).is_file():
            return candidate
    return "ffmpeg"


def to_wav_16k(src: str | Path, ffmpeg: str = "ffmpeg", out_dir: Path | None = None) -> Path:
    """转成 16kHz 单声道 s16le 的 wav。"""
    src = Path(src)
    out_dir = out_dir or Path(tempfile.mkdtemp(prefix="sv_stt_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}_16k.wav"
    proc = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            "-af", "asetpts=PTS-STARTPTS",
            "-y", str(dst),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    if proc.returncode != 0 or not dst.is_file():
        raise RuntimeError(f"ffmpeg 转换失败: {(proc.stderr or '')[:200]}")
    return dst


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        if f.getframerate() != 16000 or f.getnchannels() != 1:
            raise RuntimeError("音频不是 16k 单声道")
        data = f.readframes(f.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / f.getframerate()


async def resolve_audio_path(audio_ref: str, ffmpeg: str, max_seconds: int) -> tuple[Path, float]:
    """把 audio_url / 本地路径统一成 16k wav，返回 (路径, 时长)。"""
    from astrbot.core.utils.media_utils import MediaResolver

    loop = asyncio.get_running_loop()
    async with MediaResolver(
        audio_ref, media_type="audio", default_suffix=".wav"
    ).as_path(target_format="wav") as audio:
        wav = await loop.run_in_executor(None, to_wav_16k, str(audio.path), ffmpeg)
    dur = await loop.run_in_executor(None, duration_seconds, wav)
    if dur > max_seconds:
        raise RuntimeError(f"语音过长: {dur:.0f}s > 上限 {max_seconds}s")
    return wav, dur
