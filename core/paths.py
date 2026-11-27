"""模型路径解析：小模型随包，大模型走配置。

随包（<16MB）：
    models/silero_vad.onnx       0.6 MB  人声检测
        models/yamnet_class_map.csv  0.01 MB 类别表

外置（配置路径，缺了就跳过对应能力）：
    sensevoice_dir   SenseVoice 模型目录（model.int8.onnx + tokens.txt）
    yamnet_model     YAMNet 15.4 MB（声音事件）
    uvr_model        UVR-MDX 人声分离 63.7 MB
    roformer_root    RVC 目录（含 BS-Roformer 权重与 runtime）
"""

from __future__ import annotations

import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
BUNDLED = PKG_ROOT / "models"
PACKED = Path(r"H:\ASR_Enhance_Scripts")
LEGACY_PACKED = Path(r"H:\ASR_Enhance_Scrpits")

BUNDLE_LAYOUT = {
    "sensevoice_dir": ("sensevoice",),
    "yamnet_model": ("yamnet", "yamnet.onnx"),
    "yamnet_classes": ("yamnet", "yamnet_class_map.csv"),
    "uvr_model": ("uvr", "UVR-MDX-NET-Voc_FT.onnx"),
    "ffmpeg": ("ffmpeg", "bin", "ffmpeg.exe"),
}


def _first(paths: list[Path]) -> Path | None:
    for item in paths:
        try:
            if item and item.is_file():
                return item
        except OSError:
            continue
    return None


def bundled(name: str) -> Path | None:
    return _first([BUNDLED / name])


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


def resolve_yamnet(configured: str = "") -> Path | None:
    return _first(
        [
            Path(configured).expanduser() if configured else None,
            PACKED / "yamnet" / "yamnet.onnx",
        ]
    )


def resolve_yamnet_classes(configured: str = "") -> Path | None:
    return _first(
        [
            Path(configured).expanduser() if configured else None,
            BUNDLED / "yamnet_class_map.csv",
            PACKED / "yamnet" / "yamnet_class_map.csv",
        ]
    )


def resolve_uvr(configured: str = "") -> Path | None:
    return _first(
        [
            Path(configured).expanduser() if configured else None,
            PACKED / "uvr" / "UVR-MDX-NET-Voc_FT.onnx",
        ]
    )


def _sensevoice_ok(folder: Path) -> bool:
    return (folder / "model.int8.onnx").is_file() and (folder / "tokens.txt").is_file()


def normalize_sensevoice_dir(configured: str) -> Path | None:
    """把用户填的路径规整成真正的模型目录。

    支持三种填法：
      1) 文件夹（含 model.int8.onnx + tokens.txt）       → 直接用
      2) 文件夹（解压后外面还包了一层）                   → 往下找一层
      3) 直接选了 model.int8.onnx 文件                    → 取所在文件夹
    """
    raw = (configured or "").strip().strip('"').strip("'")
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_file():
        path = path.parent
    if not path.is_dir():
        return None
    if _sensevoice_ok(path):
        return path
    # 常见：解压后多一层同名目录
    try:
        for child in sorted(path.iterdir()):
            if child.is_dir() and _sensevoice_ok(child):
                return child
    except OSError:
        return None
    return None


def sensevoice_missing(path: Path | None) -> list[str]:
    """返回还缺哪些文件，给用户看。"""
    if path is None:
        return ["model.int8.onnx", "tokens.txt"]
    missing: list[str] = []
    if not (path / "model.int8.onnx").is_file():
        missing.append("model.int8.onnx")
    if not (path / "tokens.txt").is_file():
        missing.append("tokens.txt")
    return missing


def resolve_sensevoice_dir(configured: str = "") -> Path | None:
    normalized = normalize_sensevoice_dir(configured)
    if normalized is not None:
        return normalized
    candidates: list[Path] = []
    if configured and configured.strip():
        candidates.append(Path(configured).expanduser())
    env = os.getenv("ASTRBOT_SENSEVOICE_DIR", "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    candidates += [
        PACKED / "sensevoice",
        Path(r"H:\sensevoice"),
        Path(r"H:\Deepseek-Harness\Workspace\asr\models\sensevoice"),
    ]
    for item in candidates:
        try:
            if (item / "model.int8.onnx").is_file():
                return item
        except OSError:
            continue
    return None


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
        if key == "ffmpeg":
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
    result["vad_model"] = resolve_vad("")
    return result
