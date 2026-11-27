"""模型路径解析：小模型随包，大模型走配置。

外置（全部在模型包内，缺了就跳过对应能力）：
    vad_model        silero_vad.onnx（0.6 MB，人声检测）
    sensevoice_dir   SenseVoice 模型目录（model.int8.onnx + tokens.txt）
    yamnet_model     YAMNet 15.4 MB（声音事件）
    uvr_model        UVR-MDX 人声分离 63.7 MB
    ffmpeg          音频解码
"""

from __future__ import annotations

import os
from pathlib import Path

PACKED = Path(r"H:\ASR_Enhance_Scripts")
LEGACY_PACKED = Path(r"H:\ASR_Enhance_Scrpits")

BUNDLE_LAYOUT = {
    "sensevoice_dir": ("sensevoice",),
    "yamnet_model": ("yamnet", "yamnet.onnx"),
    "yamnet_classes": ("yamnet", "yamnet_class_map.csv"),
    "uvr_model": ("uvr", "UVR-MDX-NET-Voc_FT.onnx"),
    "ffmpeg": ("ffmpeg", "bin", "ffmpeg.exe"),
    "vad_model": ("vad", "silero_vad.onnx"),
}

def _first(paths: list[Path]) -> Path | None:
    for item in paths:
        try:
            if item and item.is_file():
                return item
        except OSError:
            continue
    return None

def resolve_vad(configured: str = "") -> Path | None:
    return _first(
        [
            Path(configured).expanduser() if configured else None,
            Path(os.getenv("ASTRBOT_SILERO_VAD", "")).expanduser() if os.getenv("ASTRBOT_SILERO_VAD") else None,
            BUNDLED / "silero_vad.onnx",
            PACKED / "silero_vad.onnx",
            Path(r"H:\Deepseek-Harness\Workspace\asr\scripts\silero_vad.onnx"),
            Path(r"H:\models\silero_vad.onnx"),
        ]
    )

def resolve_bundle(configured_root: str = "") -> dict:
    """按模型包目录推导所有外部模型路径。

    返回 {root, sensevoice_dir, yamnet_model, yamnet_classes, uvr_model, ffmpeg, missing}
    """
    candidates: list[Path] = []
    if configured_root and configured_root.strip():
        candidates.append(Path(configured_root.strip().strip('"').strip("'")).expanduser())
    candidates.append(PACKED)
    candidates.append(LEGACY_PACKED)

    root: Path | None = None
    for item in candidates:
        try:
            if (item / "sensevoice" / "model.int8.onnx").is_file() or (item / "yamnet").is_dir():
                root = item
                break
        except OSError:
            continue
    if root is None:
        root = candidates[0]

    result: dict = {"root": root, "missing": []}
    for key, parts in BUNDLE_LAYOUT.items():
        if key == "vad_model":
            options = [
                root / "vad" / "silero_vad.onnx",
                root / "silero_vad.onnx",
            ]
            path = next((x for x in options if x.exists()), options[0])
        elif key == "ffmpeg":
            # 兼容三种摆放：包根 / ffmpeg\bin / ffmpeg
            options = [
                root / "ffmpeg.exe",
                root / "ffmpeg" / "bin" / "ffmpeg.exe",
                root / "ffmpeg" / "ffmpeg.exe",
            ]
            path = next((x for x in options if x.exists()), options[0])
        else:
            path = root.joinpath(*parts)
        result[key] = path
        if not path.exists():
            result["missing"].append(str(path))

    # 内置小模型（随包）
    return result
