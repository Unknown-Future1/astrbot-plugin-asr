"""astrbot_plugin_sensevoice_asr_enhance —— 独立语音插件。

QQ 语音 → 判型（VAD / 音乐 / 音效）→ 可选分离 → SenseVoice → 文字 + 声音信息

不依赖 RVC、不依赖其它插件；所有模型 ONNX/CPU。
"""

from __future__ import annotations

import time
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Record, Reply
from astrbot.api.star import Context, Star, register

from .core import paths, provider  # noqa: F401  （注册 STT provider）
from .core.audio import find_ffmpeg
from .core.pipeline import Pipeline

__version__ = "0.2.0"

DEFAULTS = {
    "model_root": r"H:\ASR_Enhance_Scripts",
    "enable_vad": True,
    "enable_yamnet": True,
    "enable_separate": True,
    "enable_sensevoice": True,
    "trigger_mode": "always",
    "reply_with_audio": True,
    "num_threads": 0,
    "mode": "provider",
    "with_emotion": True,
    "quality_threshold": 0.5,
    "annotate_events": True,
    "silent_reply": "[这段语音里没有人声]",
    "separate_max_seconds": 0,
    "dialect_reply": "[这段语音没听清（可能是方言、口音，或背景音太重），可以卖萌表示没听懂]",
}


@register(
    "astrbot_plugin_sensevoice_asr_enhance",
    "未知の未来",
    "本地语音识别：先判声音构成，再决定分离，输出文字与声音信息，纯 CPU 无 torch",
    __version__,
)
class AsrEnhancePlugin(Star):
    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context)
        self.config = {**DEFAULTS, **(config or {})}
        self.pipeline = self._build()
        if str(self.config.get("mode", "provider")) == "provider":
            self._provision_stt()
        logger.info(
            "[ASR+] 就绪 | VAD=%s | YAMNet=%s | 分离=%s | SenseVoice=%s | 触发=%s"
            % (
                bool(self.config.get("enable_vad", True)),
                bool(self.config.get("enable_yamnet", True)),
                bool(self.config.get("enable_separate", True)),
                bool(self.config.get("enable_sensevoice", True)),
                self.config.get("trigger_mode", "always"),
            )
        )

    def _provision_stt(self) -> None:
        """确保 AstrBot 里有本插件的 STT provider，并设为默认。"""
        try:
            cfg = self.context.get_config()
            providers = list(cfg.get("provider") or [])
            changed = False

            ours = [p for p in providers if p.get("type") == provider.PROVIDER_TYPE]
            if not ours:
                entry = dict(provider.TEMPLATE)
                entry["model_root"] = str(self.config.get("model_root") or entry["model_root"])
                entry["enable"] = True
                providers.append(entry)
                ours = [entry]
                changed = True

            # 同步插件设置到 provider
            for item in ours:
                for key in (
                    "model_root", "with_emotion", "enable_vad", "enable_yamnet",
                    "enable_separate", "max_seconds", "num_threads",
                ):
                    value = self.config.get(key)
                    if value is not None and item.get(key) != value:
                        item[key] = value
                        changed = True
                if not item.get("enable", True):
                    item["enable"] = True
                    changed = True

            # 停用旧的 sensevoice provider，避免两个 STT 抢
            for item in providers:
                if item.get("type") == "sensevoice_sherpa_stt" and item.get("enable", True):
                    item["enable"] = False
                    changed = True

            stt = dict(cfg.get("provider_stt_settings") or {})
            if not stt.get("enable") or stt.get("provider_id") != provider.PROVIDER_ID:
                stt = {"enable": True, "provider_id": provider.PROVIDER_ID}
                cfg["provider_stt_settings"] = stt
                changed = True

            if changed:
                cfg["provider"] = providers
                cfg.save_config()
                logger.info(
                    "[ASR+] 已注册 STT provider %s 并设为默认（重启后生效）",
                    provider.PROVIDER_ID,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[ASR+] 自动注册 STT provider 失败: {exc}")

    def _build(self) -> Pipeline:
        cfg = self.config
        bundle = paths.resolve_bundle(str(cfg.get("model_root", "") or ""))
        self.bundle = bundle
        missing = bundle.get("missing") or []
        if missing:
            logger.warning("[ASR+] 模型包缺少 %d 个文件：%s", len(missing), missing)
        return Pipeline(
            vad_model=bundle.get("vad_model"),
            yamnet_model=bundle.get("yamnet_model"),
            yamnet_classes=bundle.get("yamnet_classes"),
            uvr_model=bundle.get("uvr_model"),
            sensevoice_dir=bundle.get("sensevoice_dir"),
            enable_vad=bool(cfg.get("enable_vad", True)),
            enable_yamnet=bool(cfg.get("enable_yamnet", True)),
            enable_separate=bool(cfg.get("enable_separate", True)),
            enable_sensevoice=bool(cfg.get("enable_sensevoice", True)),
            num_threads=int(cfg.get("num_threads", 4) or 4),
            ffmpeg=find_ffmpeg(str(bundle.get("ffmpeg") or "")),
        )

    # ---------------- 语音入口 ----------------
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_voice(self, event: AstrMessageEvent):
        # provider 模式：交给 AstrBot 的 STT 流程，插件不再自己发言
        if str(self.config.get("mode", "provider")) != "chat":
            return
        record = self._pick_record(event)
        if record is None:
            return
        if not self._should_handle(event):
            return

        try:
            path = Path(await record.convert_to_file_path())
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[ASR+] 取语音失败: {exc}")
            return

        await self._run_and_reply(event, path, show_detail=False)

    async def _run_and_reply(self, event: AstrMessageEvent, path: Path, show_detail: bool) -> None:
        work = Path("data") / "plugin_data" / "astrbot_plugin_sensevoice_asr_enhance" / "work" / f"{int(time.time())}"
        work.mkdir(parents=True, exist_ok=True)
        started = time.time()
        try:
            result = self.pipeline.run(path, work)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[ASR+] 处理失败: {exc}", exc_info=True)
            await self._send(event, f"语音处理失败：{str(exc)[:100]}")
            return
        cost = time.time() - started

        if show_detail:
            audio = result.get("audio") or {}
            speech = result.get("speech") or {}
            events = "、".join(f"{k} {v:.2f}" for k, v in (audio.get("events") or [])[:3])
            lines = [
                f"时长 {audio.get('duration')}s｜人声 {audio.get('vad_ratio') or 0:.0%}"
                f"｜音乐 {(audio.get('music') or 0):.0%}",
                f"判定：{(result.get('decision') or {}).get('path')}"
                f"（{(result.get('decision') or {}).get('reason')}）",
                f"处理：{(result.get('processing') or {}).get('method')}｜耗时 {cost:.1f}s",
                f"识别：{speech.get('text') or '（空）'}",
            ]
            if events:
                lines.insert(1, f"音效：{events}")
            await self._send(event, "\n".join(lines))
            return

        text = self.pipeline.render(
            result, with_audio=bool(self.config.get("reply_with_audio", True))
        )
        await self._send(event, text)
        event.stop_event()   # 写死：chat 模式不再触发大模型，避免重复应答

    # ---------------- 命令 ----------------
    @filter.command("asr_enhance")
    async def cmd_status(self, event: AstrMessageEvent):
        """查看语音增强状态。"""
        cfg = self.config
        p = self.pipeline
        bundle = getattr(self, "bundle", {})
        lines = [
            "语音增强 ASR+",
            f"  模型包：{bundle.get('root')}"
            + (f"（缺 {len(bundle.get('missing') or [])} 个文件）" if bundle.get("missing") else "（完整）"),
            f"  触发方式：{cfg.get('trigger_mode')}",
            f"  VAD：{'开' if p.vad else '关'}",
            f"  YAMNet：{'开' if p.enable_yamnet else '关'}"
            f"｜{'可用' if p.enhancer.events.available() else '模型缺失'}",
            f"  分离：{'UVR' if p.separator and p.separator.available() else '不可用/关闭'}",
            f"  SenseVoice：{'可用' if p.sensevoice and p.sensevoice.available() else '模型缺失'}"
            f"（{p.sensevoice.model_dir if p.sensevoice else '—'}）",
        ]
        await self._send(event, "\n".join(lines))

    @filter.command("asr_enhance_check")
    async def cmd_check(self, event: AstrMessageEvent):
        """引用一条语音，查看判型与识别结果。"""
        record = self._pick_record(event)
        if record is None:
            await self._send(event, "请引用一条语音再发 /asr_enhance_check")
            return
        try:
            path = Path(await record.convert_to_file_path())
        except Exception as exc:  # noqa: BLE001
            await self._send(event, f"取语音失败：{exc}")
            return
        await self._run_and_reply(event, path, show_detail=True)

    # ---------------- 工具 ----------------
    def _should_handle(self, event: AstrMessageEvent) -> bool:
        mode = str(self.config.get("trigger_mode", "always"))
        if mode == "at":
            return bool(event.is_at_or_wake_command)
        if mode == "reply":
            return self._quoted_record(event) is not None
        return True

    @staticmethod
    def _quoted_record(event: AstrMessageEvent):
        for comp in event.get_messages():
            if isinstance(comp, Reply) and comp.chain:
                for sub in comp.chain:
                    if isinstance(sub, Record) and getattr(sub, "file", None):
                        return sub
        return None

    def _pick_record(self, event: AstrMessageEvent):
        quoted = self._quoted_record(event)
        if quoted is not None:
            return quoted
        for comp in event.get_messages():
            if isinstance(comp, Record) and getattr(comp, "file", None):
                return comp
        return None

    async def _send(self, event: AstrMessageEvent, text: str) -> None:
        try:
            await event.send(event.plain_result(text))
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[ASR+] 发送失败: {exc}")
