"""共享转写服务：只保留 SenseVoice（Qwen3-ASR 兜底已移除）。

对外接口（供 QQ 语音与视频理解插件复用）：
  · recognize_samples(samples)
  · start() / stop() / status()

模型只从外部路径加载，插件包内不包含模型文件。
"""

from __future__ import annotations

import threading

import numpy as np

from .recognizer import SenseVoiceRecognizer, speech_quality, to_simplified


class TranscriptionService:
    """SenseVoice 识别器的持有者，同时充当「服务」的启停入口。"""

    def __init__(
        self,
        sensevoice: SenseVoiceRecognizer,
        *,
        quality_threshold: float = 0.5,
    ) -> None:
        self.sensevoice = sensevoice
        self.quality_threshold = float(quality_threshold)
        self._lock = threading.Lock()

    # ---------------- 服务控制 ----------------
    def start(self) -> dict:
        self.sensevoice._ensure()
        return {"sensevoice": self.sensevoice.loaded()}

    def stop(self) -> dict:
        with self._lock:
            unloaded = self.sensevoice.unload()
        return {"sensevoice": unloaded}

    def status(self) -> dict:
        ok, err = self.sensevoice.available()
        return {
            "sensevoice": {
                "dir": str(self.sensevoice.model_dir),
                "available": ok,
                "loaded": self.sensevoice.loaded(),
                "error": err,
            }
        }

    # ---------------- 转写 ----------------
    def recognize_samples(self, samples: np.ndarray) -> dict:
        normalized = np.asarray(samples, dtype=np.float32).reshape(-1)
        duration = normalized.size / 16000
        result = self.sensevoice.recognize(normalized)
        result["engine"] = "sensevoice"
        result["duration"] = round(duration, 2)
        return result

    def quality(self, text: str, duration: float) -> tuple[float, str]:
        return speech_quality(text, duration)


def get_transcription_service(context) -> TranscriptionService:
    """供兄弟插件（如视频理解）复用同一套已加载的识别器。"""
    for provider in context.get_all_stt_providers():
        config = getattr(provider, "provider_config", {}) or {}
        if config.get("type") == "sensevoice_sherpa_stt":
            service = getattr(provider, "transcription_service", None)
            if service is not None:
                return service
    raise RuntimeError(
        "SenseVoice 插件的 STT provider 尚未加载或未启用；"
        "请确认 provider 配置已启用并重启 AstrBot。"
    )
