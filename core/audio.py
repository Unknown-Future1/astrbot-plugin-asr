"""音频转码：QQ 语音 amr/mp3/m4a → 16k 单声道 wav。

ffmpeg 使用策略（一个规则，不做多层探测）：
    找到可用的 ffmpeg 后，统一复制到 %TEMP%\\asr_enhance_ffmpeg\\ffmpeg.exe 再执行。
原因：部分目录（比如被标成 Low 完整性级别的模型包）里的 exe
      运行时无法写入 C:，复制到临时目录（Medium）即可正常。
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import wave

PACKED = Path(r"H:\ASR_Enhance_Scripts")
LEGACY_PACKED = Path(r"H:\Sensevoice")

# 副本存放位置：插件数据目录（不会被临时目录清理），失败再退到 %TEMP%
COPY_DIRS = [
    Path("data") / "plugin_data" / "astrbot_plugin_sensevoice_asr_enhance" / "bin",
    Path(tempfile.gettempdir()) / "asr_enhance_ffmpeg",
]
_ACTIVE: str | None = None


def _first_exe(configured: str = "") -> str | None:
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    for root in (PACKED, LEGACY_PACKED):
        candidates += [
            root / "ffmpeg.exe",
            root / "ffmpeg" / "bin" / "ffmpeg.exe",
            root / "ffmpeg" / "ffmpeg.exe",
        ]
    candidates += [
        Path(r"H:\Python\tools\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
    ]
    for item in candidates:
        try:
            if item.is_file():
                return str(item)
        except OSError:
            continue
    found = shutil.which("ffmpeg")
    return found


def _probe(exe: str) -> bool:
    """用 0.2 秒静音真写一次文件，确认这个 ffmpeg 在本机可用。"""
    if not exe:
        return False
    target = Path(tempfile.gettempdir()) / f"asr_enh_probe_{abs(hash(exe)) % 10000}.wav"
    try:
        subprocess.run(
            [exe, "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "0.2",
             "-y", str(target)],
            capture_output=True, timeout=20,
        )
        return target.is_file() and target.stat().st_size > 100
    except Exception:  # noqa: BLE001
        return False
    finally:
        try:
            target.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass


def _copy_to_dir(src: Path, target_dir: Path) -> str | None:
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        dst = target_dir / "ffmpeg.exe"
        if not dst.is_file() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        return str(dst)
    except Exception:  # noqa: BLE001
        return None


def find_ffmpeg(configured: str = "") -> str:
    """返回实际可用的 ffmpeg；副本放插件数据目录，临时目录只作兜底。"""
    exe = _first_exe(configured)
    copy_path: str | None = None

    if exe:
        src = Path(exe)
        for target_dir in COPY_DIRS:
            copy_path = _copy_to_dir(src, target_dir)
            if copy_path and _probe(copy_path):
                break
            copy_path = None

    for candidate in (copy_path, exe):
        if candidate and _probe(candidate):
            return candidate

    found = shutil.which("ffmpeg")
    if found and _probe(found):
        return found

    return copy_path or exe or "ffmpeg"


def ensure_ffmpeg(configured: str = "") -> str:
    """用前自愈：缓存路径被清理/删除时自动重新复制。"""
    global _ACTIVE
    if _ACTIVE:
        try:
            if Path(_ACTIVE).is_file():
                return _ACTIVE
        except OSError:
            pass
    _ACTIVE = find_ffmpeg(configured)
    return _ACTIVE


def to_wav_16k(src: str | Path, ffmpeg: str = "ffmpeg", out_dir: Path | None = None) -> Path:
    exe = ensure_ffmpeg(ffmpeg)
    src = Path(src).resolve()
    out_dir = Path(out_dir).resolve() if out_dir else Path(tempfile.mkdtemp(prefix="asr_enh_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}_16k.wav"
    if dst.exists():
        try:
            dst.unlink()
        except OSError:
            pass
    proc = subprocess.run(
        [
            exe, "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            "-af", "asetpts=PTS-STARTPTS",
            "-f", "wav",
            "-y", str(dst),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    if proc.returncode != 0 or not dst.is_file():
        detail = (proc.stderr or "").strip()
        # ffmpeg 副本被清理掉的典型报错：No such file / Permission denied
        if "No such file" in detail or "Permission denied" in detail:
            global _ACTIVE
            _ACTIVE = None
            exe = ensure_ffmpeg(ffmpeg)
            proc = subprocess.run(
                [
                    exe, "-hide_banner", "-loglevel", "error",
                    "-i", str(src),
                    "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                    "-af", "asetpts=PTS-STARTPTS", "-f", "wav",
                    "-y", str(dst),
                ],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=120,
            )
            detail = (proc.stderr or "").strip()
        if proc.returncode != 0 or not dst.is_file():
            raise RuntimeError(f"ffmpeg 转换失败: {detail[:200]}")
    return dst


def is_wav_16k(path: str | Path) -> bool:
    try:
        with wave.open(str(path), "rb") as f:
            return f.getframerate() == 16000 and f.getnchannels() == 1 and f.getsampwidth() == 2
    except Exception:  # noqa: BLE001
        return False


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as f:
        sr = f.getframerate()
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        if f.getnchannels() > 1:
            data = data.reshape(-1, f.getnchannels()).mean(axis=1).astype(np.int16)
    return data.astype(np.float32) / 32768.0, sr


def prepare_audio(src: str | Path, ffmpeg: str, work_dir: Path) -> Path:
    """保证返回 16k 单声道 wav：本来就是就复制，不是就转码。"""
    src = Path(src).resolve()
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    if is_wav_16k(src):
        dst = work_dir / "input_16k.wav"
        if dst != src:
            shutil.copy2(src, dst)
        return dst
    return to_wav_16k(src, ffmpeg, work_dir)
