"""把 SenseVoice 注册成 AstrBot 的 STT provider。

注册时机：插件在 provider_manager.initialize() 之前被加载（core_lifecycle L267/271），
所以这里的装饰器执行后，provider_cls_map 里就有这个类型，面板可直接选。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from astrbot.api import logger
from astrbot.core.provider.entities import ProviderType
from astrbot.core.provider.provider import STTProvider
from astrbot.core.provider.register import register_provider_adapter

from . import cache, recent
from .audio import find_ffmpeg, read_wav, resolve_audio_path
from .recognizer import SenseVoiceRecognizer
from .paths import resolve_sensevoice_dir
from .transcription import TranscriptionService

PROVIDER_TYPE = "sensevoice_sherpa_stt"

DEFAULT_CONFIG = {
    "model_dir": "",
    "with_emotion": True,
    "num_threads": 6,
    "max_seconds": 180,
    "ffmpeg_path": "",
    "keep_loaded": True,
    "quality_threshold": 0.5,
    "dialect_reply": "[对方说了一段方言或口音很重的话，你没听懂，可以卖萌表示听不懂]",
}


@register_provider_adapter(
    PROVIDER_TYPE,
    "SenseVoice 本地语音识别 (sherpa-onnx)",
    provider_type=ProviderType.SPEECH_TO_TEXT,
    default_config_tmpl={
        "id": PROVIDER_TYPE,
        "type": PROVIDER_TYPE,
        "provider_type": "speech_to_text",
        "model_dir": "",
        "with_emotion": True,
        "num_threads": 6,
        "max_seconds": 180,
        "ffmpeg_path": "",
        "keep_loaded": True,
        "quality_threshold": 0.5,
        "dialect_reply": "[对方说了一段方言或口音很重的话，你没听懂，可以卖萌表示听不懂]",
                        },
)
class ProviderSenseVoiceSTT(STTProvider):
    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config, provider_settings)
        cfg = {**DEFAULT_CONFIG, **(provider_config or {})}
        self.model_dir = self._resolve_model_dir(cfg.get("model_dir", ""))
        self.with_emotion = bool(cfg.get("with_emotion", True))
        self.num_threads = int(cfg.get("num_threads", 6) or 6)
        self.max_seconds = int(cfg.get("max_seconds", 180) or 180)
        self.ffmpeg = find_ffmpeg(str(cfg.get("ffmpeg_path", "") or ""))
        self.quality_threshold = float(cfg.get("quality_threshold", 0.5) or 0.5)
        self.dialect_reply = str(
            cfg.get("dialect_reply")
            or "[对方说了一段方言或口音很重的话，你没听懂，可以卖萌表示听不懂]"
        )
        self.recognizer = SenseVoiceRecognizer(
            model_dir=self.model_dir,
            num_threads=self.num_threads,
            keep_loaded=bool(cfg.get("keep_loaded", True)),
        )
        self.transcription_service = TranscriptionService(
            self.recognizer,
            quality_threshold=self.quality_threshold,
        )
        self.set_model("sensevoice-small-int8")
        ok, err = self.recognizer.available()
        if ok:
            logger.info(f"[SenseVoice] 模型就绪: {self.model_dir}")
        else:
            logger.error(
                f"[SenseVoice] 模型不可用: {err}。"
                "请在 provider 配置里填写 model_dir，或放到 "
                "data/plugin_data/astrbot_plugin_sensevoice_asr/models/sensevoice"
            )

    @staticmethod
    def _resolve_model_dir(configured: str) -> Path:
        return resolve_sensevoice_dir(configured)

    async def initialize(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self.recognizer._ensure)
            logger.info("[SenseVoice] 模型加载完成")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[SenseVoice] 模型加载失败: {exc}")

    async def get_text(self, audio_url: str) -> str:
        loop = asyncio.get_running_loop()
        cached = cache.get(str(audio_url))
        if cached is not None:
            logger.debug(f"[SenseVoice] 命中缓存: {audio_url[:60]}")
            return self._render(cached, cached.get("duration") or 0.0)
        try:
            wav, dur = await resolve_audio_path(audio_url, self.ffmpeg, self.max_seconds)
            samples = await loop.run_in_executor(None, read_wav, wav)
            result = await loop.run_in_executor(
                None, self.transcription_service.recognize_samples, samples
            )
            result["duration"] = dur
            cache.put(str(audio_url), result)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[SenseVoice] 转写失败: {exc}")
            return ""

        return self._render(result, dur)

    def _render(self, result: dict, dur: float) -> str:
        text = (result.get("text") or "").strip()
        if not text:
            logger.info("[SenseVoice] 未识别到语音（纯音乐/静音）")
            return ""

        quality = float(result.get("quality") or 0.0)
        if quality < self.quality_threshold:
            logger.info(
                f"[SenseVoice] 疑似方言/失真（quality={quality}, {result.get('quality_reason')}）"
                f" 原文={text[:30]}"
            )
            recent.put_recent("dialect", self.dialect_reply)
            return self.dialect_reply

        if self.with_emotion and result.get("emo"):
            rendered = f"（{result['emo']}）{text}"
            logger.info(f"[SenseVoice] {dur:.1f}s → {result['emo']} | {text[:40]}")
        else:
            rendered = text
            logger.info(f"[SenseVoice] {dur:.1f}s → {text[:40]}")
        recent.put_recent("ok", rendered)
        return rendered
