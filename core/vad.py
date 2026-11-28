"""人声检测（silero VAD）。"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np


class Vad:
    def __init__(self, model: Path | None, threshold: float = 0.35):
        self.model = Path(model) if model else None
        self.threshold = threshold
        self._detector = None
        self._buffer_seconds = 60.0
        self._capacity = 0.0
        self._lock = threading.Lock()

    def available(self) -> bool:
        return bool(self.model and self.model.is_file())

    def _ensure(self):
        need = max(60.0, self._buffer_seconds + 5)
        if self._detector is not None and need <= self._capacity:
            return self._detector
        import sherpa_onnx

        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(self.model)
        cfg.silero_vad.threshold = float(self.threshold)
        cfg.silero_vad.min_silence_duration = 0.3
        cfg.silero_vad.min_speech_duration = 0.2
        cfg.sample_rate = 16000
        cfg.num_threads = 1
        # 按历史最长音频分配，只增不减；复用同一个检测器
        self._capacity = max(self._capacity, need)
        self._detector = sherpa_onnx.VoiceActivityDetector(
            cfg, buffer_size_in_seconds=int(self._capacity)
        )
        return self._detector

    def segments(self, samples: np.ndarray, sr: int) -> list[dict]:
        """返回人声段 [{t0, t1}]，样本需 16k。"""
        if not self.available():
            return []
        import librosa

        audio = np.asarray(samples, dtype=np.float32)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        with self._lock:
            return self._segments_locked(audio)

    def _segments_locked(self, audio: np.ndarray) -> list[dict]:
        self._buffer_seconds = len(audio) / 16000
        vad = self._ensure()
        vad.reset()   # 复用前清空上一次的状态
        out: list[dict] = []
        step = 8000
        for i in range(0, audio.size, step):
            vad.accept_waveform(audio[i : i + step])
            while not vad.empty():
                seg = vad.front
                start = len(seg.samples)
                out.append(
                    {
                        "t0": round(seg.start / 16000, 3),
                        "t1": round((seg.start + start) / 16000, 3),
                    }
                )
                vad.pop()
        vad.flush()
        while not vad.empty():
            seg = vad.front
            start = len(seg.samples)
            out.append(
                {
                    "t0": round(seg.start / 16000, 3),
                    "t1": round((seg.start + start) / 16000, 3),
                }
            )
            vad.pop()
        return out

    def speech_ratio(self, samples: np.ndarray, sr: int) -> tuple[float, list[dict]]:
        segs = self.segments(samples, sr)
        duration = max(len(samples) / sr, 1e-6)
        total = sum(max(0.0, s["t1"] - s["t0"]) for s in segs)
        return round(min(1.0, total / duration), 3), segs
