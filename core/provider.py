"""把本插件注册成 AstrBot 的 STT provider。

这样语音转写会作为「用户消息」进入对话上下文，由后台大模型决定怎么回，
而不是插件自己往群里发文字。

注册时机：插件在 provider_manager.initialize() 之前加载，装饰器执行后
provider_cls_map 里就有这个类型；插件 __init__ 里会确保 cmd_config 中存在
对应 provider 条目并设为默认 STT，做到开箱即用。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from astrbot.api import logger
from astrbot.core.provider.entities import ProviderType
from astrbot.core.provider.provider import STTProvider
from astrbot.core.provider.register import register_provider_adapter

from . import paths
from .pipeline import Pipeline

PROVIDER_TYPE = "asr_enhance_stt"
PROVIDER_ID = "asr-enhance-local"

TEMPLATE = {
    "id": PROVIDER_ID,
    "type": PROVIDER_TYPE,
    "provider_type": "speech_to_text",
    "enable": True,
    "model_root": r"H:\ASR_Enhance_Scripts",
    "with_emotion": True,
    "enable_vad": True,
    "enable_yamnet": True,
    "enable_separate": True,
    "max_seconds": 180,
    "num_threads": 0,
    "quality_threshold": 0.5,
    "annotate_events": True,
    "silent_reply": "[这段语音里没有人声]",
    "separate_max_seconds": 0,
    "dialect_reply": "[这段语音没听清（可能是方言、口音，或背景音太重），可以卖萌表示没听懂]",
}


@register_provider_adapter(
    PROVIDER_TYPE,
    "Sensevoice ASR语音转写增强（VAD/分离 + SenseVoice）",
    provider_type=ProviderType.SPEECH_TO_TEXT,
    default_config_tmpl=dict(TEMPLATE),
)
class ProviderAsrEnhance(STTProvider):
    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config, provider_settings)
        cfg = {**TEMPLATE, **(provider_config or {})}
        self.model_root = str(cfg.get("model_root") or "")
        self.with_emotion = bool(cfg.get("with_emotion", False))
        self.max_seconds = int(cfg.get("max_seconds", 180) or 180)
        self.num_threads = int(cfg.get("num_threads", 0) or 0)
        self.quality_threshold = float(cfg.get("quality_threshold", 0.5) if cfg.get("quality_threshold") is not None else 0.5)
        self.annotate_events = bool(cfg.get("annotate_events", True))
        self.silent_reply = str(cfg.get("silent_reply") or "[这段语音里没有人声]")
        self.separate_max_seconds = float(cfg.get("separate_max_seconds", 0) or 0)
        self.dialect_reply = str(
            cfg.get("dialect_reply")
            or "[这段语音没听清（可能是方言、口音，或背景音太重），可以卖萌表示没听懂]"
        )
        self.enable_vad = bool(cfg.get("enable_vad", True))
        self.enable_yamnet = bool(cfg.get("enable_yamnet", True))
        self.enable_separate = bool(cfg.get("enable_separate", True))
        self._pipeline: Pipeline | None = None
        self.set_model("asr-enhance-local")

    # ---------------- 生命周期 ----------------
    def _ensure_pipeline(self) -> Pipeline:
        if self._pipeline is not None:
            return self._pipeline
        bundle = paths.resolve_bundle(self.model_root)
        missing = bundle.get("missing") or []
        if missing:
            logger.warning("[ASR+] provider 模型包缺文件: %s", missing)
        self._pipeline = Pipeline(
            vad_model=bundle.get("vad_model"),
            yamnet_model=bundle.get("yamnet_model"),
            yamnet_classes=bundle.get("yamnet_classes"),
            uvr_model=bundle.get("uvr_model"),
            sensevoice_dir=bundle.get("sensevoice_dir"),
            enable_vad=self.enable_vad,
            enable_yamnet=self.enable_yamnet,
            enable_separate=self.enable_separate,
            enable_sensevoice=True,
            num_threads=self.num_threads,
            ffmpeg=str(bundle.get("ffmpeg") or ""),
            quality_threshold=self.quality_threshold,
            separate_max_seconds=self.separate_max_seconds,
        )
        return self._pipeline

    async def initialize(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            pipeline = await loop.run_in_executor(None, self._ensure_pipeline)
            if pipeline.sensevoice and pipeline.sensevoice.available():
                await loop.run_in_executor(None, pipeline.sensevoice._ensure)
                logger.info(
                    "[ASR+] STT provider 就绪 | SenseVoice=%s | 分离=%s | YAMNet=%s | ffmpeg=%s",
                    pipeline.sensevoice.model_dir,
                    "UVR" if pipeline.separator and pipeline.separator.available() else "关",
                    pipeline.enable_yamnet,
                    pipeline.ffmpeg,
                )
            else:
                logger.error("[ASR+] SenseVoice 模型缺失，检查设置里的模型包目录")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[ASR+] STT provider 初始化失败: {exc}")

    # ---------------- 识别 ----------------
    async def get_text(self, audio_url: str) -> str:
        loop = asyncio.get_running_loop()
        try:
            path = await self._resolve_path(audio_url)
            work = Path("data") / "plugin_data" / "astrbot_plugin_sensevoice_asr_enhance" / "stt" / str(
                abs(hash(str(audio_url))) % 10_000_000
            )
            pipeline = await loop.run_in_executor(None, self._ensure_pipeline)
            result = await loop.run_in_executor(None, pipeline.run, path, work)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[ASR+] 转写失败: {exc}")
            return ""

        speech = result.get("speech") or {}
        text = (speech.get("text") or "").strip()
        if not text:
            logger.info("[ASR+] 没听到人声（纯音乐或静音）")
            return self.silent_reply

        quality = float(speech.get("quality") or 0.0)
        if quality < self.quality_threshold:
            logger.info(
                "[ASR+] 识别可信度低（q=%.2f < %.2f，%s），返回方言提示",
                quality,
                self.quality_threshold,
                (result.get("decision") or {}).get("path"),
            )
            return self.dialect_reply
        if self.annotate_events:
            text = self._annotate(result, text)

        if self.with_emotion and speech.get("emotion"):
            text = f"（{speech['emotion']}）{text}"
        logger.info(
            "[ASR+] %.1fs → %s（%s）",
            (result.get("audio") or {}).get("duration") or 0,
            text[:50],
            (result.get("decision") or {}).get("path"),
        )
        return text

    def _annotate(self, result: dict, text: str) -> str:
        """把笑声/掌声/咳嗽等社交事件按时间插到句子后面。

        · 只标社交类（噪声类不进文本）
        · 置信度 ≥0.35 直接写「（笑声 5.2s）」
        · 0.15~0.35 写「（疑似笑声 5.2s）」
        """
        from .events import SOCIAL_TAGS

        audio = result.get("audio") or {}
        timeline = audio.get("timeline") or []
        social = [e for e in timeline if e.get("label") in SOCIAL_TAGS]
        if not social:
            return text

        def mark(event: dict) -> str:
            label = event["label"]
            if float(event.get("conf") or 0) < 0.35:
                label = "疑似" + label
            return f"（{label} {event['t0']:.1f}s）"

        sentences = (result.get("speech") or {}).get("sentences") or []
        if not sentences:
            suffix = "".join(mark(e) for e in social[:5])
            return f"{text}{suffix}"

        parts: list[str] = []
        used: set[int] = set()
        for sent in sentences:
            marks = []
            for index, event in enumerate(social):
                if index in used:
                    continue
                if sent["t0"] - 0.5 <= event["t0"] <= sent["t1"] + 0.5:
                    marks.append(mark(event))
                    used.add(index)
            parts.append(sent["text"] + "".join(marks))
        # 没落到任何句子的事件，追加到末尾
        tail = "".join(
            mark(e) for index, e in enumerate(social) if index not in used
        )
        return "".join(parts) + tail

    async def _resolve_path(self, audio_url: str) -> Path:
        candidate = Path(str(audio_url))
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            pass
        from astrbot.core.utils.media_utils import MediaResolver

        async with MediaResolver(
            str(audio_url), media_type="audio", default_suffix=".wav"
        ).as_path(target_format="wav") as audio:
            return Path(audio.path)
