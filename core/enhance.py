"""Sensevoice 语音转写增强入口：判型 → 选路径 → 产出最适合 ASR 的音轨。

默认不处理（none）。只有判型命中才动手：
    vad 无人声          → skip（不送 ASR）
    音乐主导 + 有人声    → separate（UVR / BS-Roformer）
    """

from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np

from .events import SoundEvents
from .separate import Separator
from .vad import Vad


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as f:
        sr = f.getframerate()
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        if f.getnchannels() > 1:
            data = data.reshape(-1, f.getnchannels()).mean(axis=1).astype(np.int16)
    return data.astype(np.float32) / 32768.0, sr


class Enhancer:
    def __init__(
        self,
        vad_model: Path | None = None,
        yamnet_model: Path | None = None,
        yamnet_classes: Path | None = None,
        uvr_model: Path | None = None,
        enable_separate: bool = True,
        enable_yamnet: bool = True,
        num_threads: int = 4,
    ):
        self.vad = Vad(vad_model)
        self.events = SoundEvents(yamnet_model, yamnet_classes)
        self.separator = Separator(uvr_model) if enable_separate else None
        self.enable_yamnet = enable_yamnet

    def analyze(self, samples: np.ndarray, sr: int) -> dict:
        started = time.time()
        ratio, segments = self.vad.speech_ratio(samples, sr)
        profile: dict = {
            "duration": round(len(samples) / sr, 2),
            "vad_ratio": ratio,
            "vad_segments": len(segments),
        }
        events = self.events.analyze(samples, sr) if self.enable_yamnet else {"ok": False}
        if events.get("ok"):
            profile.update(
                {
                    "music": events["music"],
                    "speech": events["speech"],
                    "events": events["events"],
                    "timeline": events.get("timeline") or [],
                }
            )
        profile["seconds"] = round(time.time() - started, 2)
        return profile

    @staticmethod
    def decide(profile: dict) -> dict:
        ratio = profile.get("vad_ratio", 0.0)
        music = profile.get("music", 0.0)
        speech = profile.get("speech", 0.0)
        events = dict(profile.get("events") or [])

        if ratio < 0.02:
            return {"path": "skip", "reason": f"VAD 未检测到人声（{ratio:.0%}）"}

        if music >= 0.5 and speech < 0.5 and ratio >= 0.1:
            return {"path": "separate", "reason": f"人声与音乐混合（音乐 {music:.0%} / 语音 {speech:.0%}，VAD {ratio:.0%}）"}

        return {"path": "none", "reason": f"声音构成正常（VAD {ratio:.0%}）"}
