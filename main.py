"""astrbot_plugin_sensevoice_asr —— 本地 SenseVoice 语音识别插件。

用法（显式触发，不做自动回声）：
  群聊：引用一条语音 + @机器人 + /asr
  私聊：引用一条语音 + /asr

作用：
  · 向 AstrBot 注册 STT provider（类型 sensevoice_sherpa_stt）
  · QQ 语音转写（SenseVoice），低质量时回听不懂
  · 供视频理解插件复用同一套模型
  · 模型不打包进插件，按配置路径加载/卸载

注意：AstrBot 的 STT 在 preprocess 阶段就把 Record 换成了文字，
所以指令不能靠找语音组件，改为匹配 provider 登记的「最近一次转写」。
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Record
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core import sp

from .core import recent
from .core.provider import DEFAULT_CONFIG, PROVIDER_TYPE  # 导入即完成 provider 注册
from .core.transcription import get_transcription_service

__version__ = "0.8.0"

USAGE = (
    "用法（三种都行）：\n"
    "· 群聊：引用一条语音 + @机器人 + /asr（或直接 asr）\n"
    "· 私聊：引用一条语音 + /asr（或直接 asr）"
)

BARE_KEYWORDS = {"asr", "听一下", "转写", "转文字", "语音转文字"}
WAKE_WORDS = ("大肥鱼", "小鲸鱼", "小鲸", "溟月")


@register(
    "astrbot_plugin_sensevoice_asr",
    "未知の未来",
    "本地 SenseVoice 语音识别：QQ 语音转文字，支持情绪标注，零 API Key",
    __version__,
)
class SenseVoiceAsrPlugin(Star):
    def __init__(self, context: Context) -> None:
        super().__init__(context)
        self.config = self.context.get_config() or {}
        self._last_outgoing: tuple[float, Record] | None = None

    # ---------------- 首次启用自动设默认 provider ----------------
    async def initialize(self) -> None:
        marker = (
            StarTools.get_data_dir("astrbot_plugin_sensevoice_asr")
            / "default_stt_configured"
        )
        if marker.is_file():
            return
        try:
            config = self.context.get_config()
            providers = config.setdefault("provider", [])
            provider_cfg = next(
                (item for item in providers if item.get("type") == PROVIDER_TYPE),
                None,
            )
            if provider_cfg is None:
                provider_cfg = {
                    "id": PROVIDER_TYPE,
                    "type": PROVIDER_TYPE,
                    "provider_type": "speech_to_text",
                    "enable": True,
                    **DEFAULT_CONFIG,
                }
                providers.append(provider_cfg)
            else:
                provider_cfg["enable"] = True

            provider_id = str(provider_cfg.get("id") or PROVIDER_TYPE)
            provider_cfg["id"] = provider_id
            stt_settings = config.setdefault("provider_stt_settings", {})
            stt_settings["enable"] = True
            stt_settings["provider_id"] = provider_id

            await config.save_config_async()
            await sp.put_async(
                key="curr_provider_stt",
                value=provider_id,
                scope="global",
                scope_id="global",
            )
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("configured\n", encoding="utf-8")
            logger.info(f"[SenseVoice] 已自动设为默认 STT provider: {provider_id}")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[SenseVoice] 自动设置默认 STT provider 失败: {exc}")

    # ---------------- 配置与服务 ----------------
    @property
    def cfg(self) -> dict:
        return {**DEFAULT_CONFIG, **(self.config or {})}

    def _service(self):
        """拿到 provider 里的转写服务（含两个模型）。"""
        return get_transcription_service(self.context)

    async def _ensure_service_loaded(self):
        service = self._service()
        st = service.status()
        if st["sensevoice"]["loaded"]:
            return service
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, service.start)
        return service

    # ---------------- 文本整形 ----------------
    @staticmethod
    def _pretty(text: str) -> str:
        """「（情绪）正文」→「正文（情绪）」。"""
        m = re.match(r"^（(.+?)）(.*)$", (text or "").strip())
        if not m:
            return (text or "").strip()
        emo, body = m.group(1), m.group(2).strip()
        return f"{body}（{emo}）" if body else (text or "").strip()

    def _dialect_text(self) -> str:
        return str(self.cfg.get("dialect_echo") or "这段是方言吧，我没太听懂~")

    # ---------------- 外部 STT 结果回收 ----------------
    @staticmethod
    def _raw_voice_text(event: AstrMessageEvent) -> tuple[bool, str]:
        raw = getattr(getattr(event, "message_obj", None), "raw_message", None)
        segments = raw.get("message") if isinstance(raw, dict) else getattr(raw, "message", None)
        if not isinstance(segments, list):
            return False, ""
        has_voice = False
        texts: list[str] = []
        for seg in segments:
            if isinstance(seg, dict):
                kind = str(seg.get("type") or "").lower()
                data = seg.get("data") or {}
            else:
                kind = str(getattr(seg, "type", "") or "").lower()
                data = getattr(seg, "data", {}) or {}
            if kind in {"record", "audio", "voice"}:
                has_voice = True
            elif kind == "text":
                texts.append(str(data.get("text") or "") if isinstance(data, dict) else "")
        return has_voice, "".join(texts)

    async def _capture_external_stt(self, event: AstrMessageEvent) -> bool:
        has_voice, original = self._raw_voice_text(event)
        if not has_voice:
            return False
        try:
            provider = await self.context.get_using_stt_provider_async(
                event.unified_msg_origin
            )
        except Exception:  # noqa: BLE001
            return False
        if (getattr(provider, "provider_config", {}) or {}).get("type") == PROVIDER_TYPE:
            return False
        text = (event.get_message_str() or "").strip()
        if original:
            text = text.replace(original, "", 1).strip()
        text = re.sub(r"\[[^\]]*\]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return False
        recent.put_recent("ok", text)
        return True

    # ---------------- 裸词触发 ----------------
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def bare_asr(self, event: AstrMessageEvent):
        """引用语音 + @机器人 + 「asr」也能触发，免去斜杠。"""
        if await self._capture_external_stt(event):
            return
        if event.is_at_or_wake_command:
            return

        msg = (event.get_message_str() or "").strip()
        if not msg:
            return
        entry = recent.take_recent(msg)
        if not entry:
            return

        rest = msg.replace(entry["text"], " ")
        for w in WAKE_WORDS:
            rest = rest.replace(w, " ")
        rest = re.sub(r"\[[^\]]*\]", " ", rest)
        rest = re.sub(r"\s+", " ", rest).strip(" ，,。.！!？?~").lstrip("/").lower()
        if rest not in BARE_KEYWORDS:
            return

        if entry["kind"] == "dialect":
            yield event.plain_result(self._dialect_text())
            return
        text = self._pretty(entry["text"])
        if not text:
            yield event.plain_result("没听到人声（纯音乐或静音）")
            return
        yield event.plain_result(f'这句话说："{text}"')

    # ---------------- 出站语音捕获（回读自己的 TTS） ----------------
    @filter.on_decorating_result()
    async def capture_outgoing_voice(self, event: AstrMessageEvent):
        try:
            result = event.get_result()
            if not result or not getattr(result, "chain", None):
                return
            for comp in result.chain:
                if isinstance(comp, Record) and getattr(comp, "file", None):
                    self._last_outgoing = (time.time(), comp)
                    return
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[SenseVoice] 出站语音捕获跳过: {exc}")

    async def _transcribe_outgoing(self) -> dict | None:
        if not self._last_outgoing:
            return None
        ts, comp = self._last_outgoing
        if time.time() - ts > 3600:
            return None
        try:
            path = await comp.convert_to_file_path()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[SenseVoice] 出站语音路径解析失败: {exc}")
            return None

        from .core.audio import find_ffmpeg, read_wav, resolve_audio_path

        try:
            service = await self._ensure_service_loaded()
            wav, dur = await resolve_audio_path(
                str(path),
                find_ffmpeg(str(self.cfg.get("ffmpeg_path") or "")),
                int(self.cfg.get("max_seconds", 180) or 180),
            )
            loop = asyncio.get_running_loop()
            samples = await loop.run_in_executor(None, read_wav, wav)
            result = await loop.run_in_executor(
                None, service.recognize_samples, samples
            )
            result["duration"] = dur
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[SenseVoice] 出站语音转写失败: {exc}")
            return None

    # ---------------- 指令 ----------------
    @filter.command("asr", alias={"听一下", "转写"})
    async def cmd_asr(self, event: AstrMessageEvent):
        """转写你引用的那条语音消息（群聊需 @机器人）。"""
        await self._capture_external_stt(event)
        entry = recent.take_recent(event.get_message_str())
        fallback = False
        if not entry:
            entry = recent.peek_latest()
            fallback = entry is not None

        if not entry:
            outgoing = await self._transcribe_outgoing()
            if outgoing is None:
                yield event.plain_result("没找到要转写的语音。\n" + USAGE)
                return
            text = self._pretty(outgoing.get("text") or "")
            if not text:
                yield event.plain_result("那条语音里没听到人声（可能只有音乐）")
                return
            yield event.plain_result(f'我刚说的是："{text}"')
            return

        if entry["kind"] == "dialect":
            yield event.plain_result(self._dialect_text())
            return

        text = self._pretty(entry["text"])
        if not text:
            yield event.plain_result("没听到人声（纯音乐或静音）")
            return
        suffix = "（最近一条语音）" if fallback else ""
        yield event.plain_result(f'这句话说："{text}"{suffix}')

    @filter.command("asr_status")
    async def cmd_status(self, event: AstrMessageEvent):
        """查看模型路径与加载状态。"""
        try:
            service = self._service()
        except Exception as exc:  # noqa: BLE001
            yield event.plain_result(f"provider 未就绪：{exc}")
            return

        sv = service.status()["sensevoice"]
        lines = [
            "SenseVoice",
            f"  路径：{sv['dir']}",
            f"  文件：{'完整' if sv['available'] else '缺失'}｜加载：{'是' if sv['loaded'] else '否'}",
        ]
        if not sv["available"] and sv["error"]:
            lines.append(f"  错误：{sv['error']}")
        yield event.plain_result("\n".join(lines))

    @filter.command("asr_start")
    async def cmd_start(self, event: AstrMessageEvent):
        """按配置路径加载模型。"""
        try:
            service = self._service()
        except Exception as exc:  # noqa: BLE001
            yield event.plain_result(f"provider 未就绪：{exc}")
            return
        loop = asyncio.get_running_loop()
        info = await loop.run_in_executor(None, service.start)
        yield event.plain_result(
            f"SenseVoice：{'已加载' if info['sensevoice'] else '加载失败'}"
        )

    @filter.command("asr_stop")
    async def cmd_stop(self, event: AstrMessageEvent):
        """卸载模型，释放内存。"""
        try:
            service = self._service()
        except Exception as exc:  # noqa: BLE001
            yield event.plain_result(f"provider 未就绪：{exc}")
            return
        info = service.stop()
        yield event.plain_result(
            f"SenseVoice：{'已卸载' if info['sensevoice'] else '本来就没加载'}"
        )
