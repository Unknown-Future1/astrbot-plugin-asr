"""模型目录解析：插件只存路径，不打包模型。

查找顺序（先命中先用）：
  1. 配置里显式填写的目录
  2. 环境变量（ASTRBOT_SENSEVOICE_DIR / ASTRBOT_QWEN3_ASR_DIR）
  3. AstrBot 插件数据目录 data/plugin_data/<plugin>/models/<name>
  4. 本机开发目录（仅当存在时）
"""

from __future__ import annotations

import os
from pathlib import Path

PLUGIN_NAME = "astrbot_plugin_sensevoice_asr"

SENSEVOICE_ENV = "ASTRBOT_SENSEVOICE_DIR"


def _plugin_data_root() -> Path:
    return Path("data") / "plugin_data" / PLUGIN_NAME / "models"


def _first_ready(candidates: list[Path], marker: str) -> Path:
    for candidate in candidates:
        try:
            if candidate and (candidate / marker).is_file():
                return candidate
        except OSError:
            continue
    return candidates[0] if candidates else Path("")


def resolve_sensevoice_dir(configured: str = "") -> Path:
    """SenseVoice 目录（需要 model.int8.onnx + tokens.txt）。"""
    candidates: list[Path] = []
    if configured and configured.strip():
        candidates.append(Path(configured).expanduser())
    env = os.getenv(SENSEVOICE_ENV, "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(_plugin_data_root() / "sensevoice")
    candidates.append(Path(__file__).resolve().parents[2] / "models" / "sensevoice")
    return _first_ready(candidates, "model.int8.onnx")
