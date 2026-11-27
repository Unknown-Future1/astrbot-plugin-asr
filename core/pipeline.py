"""独立的人声流水线：音频 → 判型 → 预处理 → 识别 → 结构化结果。

不依赖任何其它插件或 RVC：
  · VAD / YAMNet / UVR 全部 ONNX + sherpa-onnx（CPU）
  · SenseVoice 模型由用户放好，路径可配
"""

from __future__ import annotations

import time
from pathlib import Path

from .audio import ensure_ffmpeg, load_wav, prepare_audio
from .enhance import Enhancer
from .sensevoice import SenseVoice
from .separate import Separator
from .vad import Vad


class Pipeline:
    def __init__(
        self,
        vad_model: Path | None = None,
        yamnet_model: Path | None = None,
        yamnet_classes: Path | None = None,
        uvr_model: Path | None = None,
        sensevoice_dir: Path | None = None,
        *,
        enable_vad: bool = True,
        enable_yamnet: bool = True,
        enable_separate: bool = True,
        enable_sensevoice: bool = True,
        num_threads: int = 0,   # 0 = 按核心数自动
        ffmpeg: str = "ffmpeg",
        quality_threshold: float = 0.5,
        rescue_separation: bool = True,
        separate_max_seconds: float = 0.0,   # 0 = 不限制
    ):
        import os

        if num_threads and num_threads > 0:
            self.num_threads = int(num_threads)
        else:
            cores = os.cpu_count() or 4
            self.num_threads = max(2, min(6, cores // 2))
        num_threads = self.num_threads

        self.ffmpeg = ensure_ffmpeg(ffmpeg)
        self.quality_threshold = float(quality_threshold)
        self.rescue_separation = bool(rescue_separation)
        self.separate_max_seconds = float(separate_max_seconds)
        self.vad = Vad(vad_model) if enable_vad else None
        self.enable_yamnet = enable_yamnet
        self.enhancer = Enhancer(
            vad_model=vad_model,
            yamnet_model=yamnet_model,
            yamnet_classes=yamnet_classes,
            num_threads=num_threads,
            enable_separate=enable_separate,
            enable_yamnet=enable_yamnet,
        )
        self.separator = Separator(uvr_model, num_threads=num_threads) if enable_separate else None
        self.sensevoice = SenseVoice(sensevoice_dir, num_threads=num_threads) if enable_sensevoice else None

    # ---------------- 主入口 ----------------
    def run(self, wav_path: Path, work_dir: Path) -> dict:
        started = time.time()
        work_dir.mkdir(parents=True, exist_ok=True)
        wav_path = prepare_audio(wav_path, self.ffmpeg, work_dir)
        samples, sr = load_wav(wav_path)
        profile = self.enhancer.analyze(samples, sr)
        plan = self.enhancer.decide(profile)

        prepared = {"wav": wav_path, "method": "none", "seconds": 0.0}
        duration = len(samples) / sr
        if (
            plan["path"] == "separate"
            and self.separate_max_seconds > 0
            and duration > self.separate_max_seconds
        ):
            plan = dict(
                plan,
                path="none",
                reason=f"音频 {duration:.0f}s 超过分离上限 {self.separate_max_seconds:.0f}s，不自动分离",
            )

        if plan["path"] == "separate" and self.separator:
            result = self.separator.separate(samples, sr, work_dir)
            if result.get("ok"):
                prepared = {"wav": result["wav"], "method": result["method"],
                            "seconds": result.get("seconds", 0.0)}
            else:
                plan = dict(plan, fallback=result.get("error", ""))
        elif plan["path"] == "skip":
            prepared = {"wav": None, "method": "skip", "seconds": 0.0}

        speech = {"text": "", "emotion": None, "event": None, "quality": 0.0}
        if prepared["wav"] is not None and self.sensevoice and self.sensevoice.available():
            proc_samples, proc_sr = load_wav(prepared["wav"])
            speech = self.sensevoice.transcribe(proc_samples, proc_sr)

        # 分离后如果反而没识别出文字，退回原始音频再试一次
        if (
            not (speech.get("text") or "").strip()
            and prepared.get("method") in ("uvr",)
            and prepared.get("wav") != wav_path
        ):
            fallback = self.sensevoice.transcribe(samples, sr)
            if (fallback.get("text") or "").strip():
                speech = fallback
                prepared = dict(prepared, method="none", fallback="empty_after_separation")

        # 低质量抢救：识别不可信、又有明显音乐时，分离后再试一次
        music_ratio = float((profile.get("music") or 0.0))
        if (
            self.rescue_separation
            and plan.get("path") != "separate"
            and float(speech.get("quality") or 0) < self.quality_threshold
            and music_ratio >= 0.4
            and self.separator is not None
            and self.separator.available()
            and (
                self.separate_max_seconds <= 0
                or (len(samples) / sr) <= self.separate_max_seconds
            )
        ):
            rescue = self.separator.separate(samples, sr, work_dir / "rescue")
            if rescue.get("ok"):
                rs, rr = load_wav(rescue["wav"])
                retried = self.sensevoice.transcribe(rs, rr)
                if float(retried.get("quality") or 0) > float(speech.get("quality") or 0):
                    speech = retried
                    prepared = {
                        "wav": rescue["wav"],
                        "method": "uvr",
                        "seconds": rescue.get("seconds", 0.0),
                        "rescue": True,
                    }
                    plan = dict(plan, path="separate", reason="低质量，分离抢救")

        return {
            "ok": True,
            "audio": {
                "duration": profile.get("duration"),
                "vad_ratio": profile.get("vad_ratio"),
                "vad_segments": profile.get("vad_segments"),
                "music": profile.get("music"),
                "speech": profile.get("speech"),
                "events": profile.get("events") or [],
                "timeline": profile.get("timeline") or [],
            },
            "decision": plan,
            "processing": prepared,
            "speech": speech,
            "seconds": round(time.time() - started, 2),
        }

    # ---------------- 输出文本 ----------------
    @staticmethod
    def render(result: dict, with_audio: bool = True) -> str:
        speech = result.get("speech") or {}
        text = (speech.get("text") or "").strip()
        if text:
            line = f"（{speech['emotion']}）{text}" if speech.get("emotion") else text
        else:
            # 没有人声（纯音乐/静音）与“听不清”区分开
            line = "没听到人声（纯音乐或静音）"
        lines = [line]
        if not with_audio:
            return lines[0]
        audio = result.get("audio") or {}
        bits = []
        if audio.get("music") is not None:
            bits.append(f"音乐 {audio['music']:.0%}")
        if audio.get("vad_ratio") is not None:
            bits.append(f"人声 {audio['vad_ratio']:.0%}")
        if audio.get("events"):
            bits.append("音效 " + "、".join(f"{k}" for k, _ in audio["events"][:3]))
        method = (result.get("processing") or {}).get("method", "none")
        if method not in ("none", "skip"):
            bits.append(f"处理 {method}")
        if bits:
            lines.append("｜".join(bits))
        return "\n".join(lines)
