"""SenseVoice 识别封装（sherpa-onnx，CPU，不需要 torch）。

输出：文本 + 情绪 + 事件（Speech/Music/BGM 等）+ 质量分。
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np

SPECIAL = re.compile(r"<\|[^|]*\|>")
SENT_END = {"。", "！", "？", "；", "…", ".", "!", "?", ";", "\n"}
SOFT = {"，", ",", "、"}
VALID = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")

EMO_MAP = {
    "HAPPY": "开心", "SAD": "悲伤", "ANGRY": "生气", "NEUTRAL": "平静",
    "FEARFUL": "害怕", "DISGUSTED": "厌恶", "SURPRISED": "惊讶",
    "EMO_UNKNOWN": "未知", "OTHER": "其他",
}

FUNC_WORDS = re.compile(r"[的了是我你他她它不在有这那就都也还和吧呢啊嘛哦么什怎吗]")
CJK = re.compile(r"[\u4e00-\u9fff]")


def speech_quality(text: str, duration: float) -> float:
    chars = CJK.findall(text)
    n = len(chars)
    if n == 0:
        return 0.0
    cps = n / max(duration, 0.3)
    speed = 1.0 if 1.5 <= cps <= 8.0 else 0.25
    uniq = len(set(chars)) / n
    diversity = 1.0 if uniq >= 0.42 else 0.3
    func_ratio = len(FUNC_WORDS.findall(text)) / n
    if func_ratio >= 0.045:
        function_score = 1.0
    elif func_ratio >= 0.025:
        function_score = 0.7
    else:
        function_score = 0.35
    return round(speed * diversity * function_score, 3)


class SenseVoice:
    def __init__(self, model_dir: Path | None, num_threads: int = 4):
        self.model_dir = Path(model_dir) if model_dir else None
        self.num_threads = num_threads
        self._rec = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        return bool(
            self.model_dir
            and (self.model_dir / "model.int8.onnx").is_file()
            and (self.model_dir / "tokens.txt").is_file()
        )

    def _ensure(self):
        if self._rec is not None:
            return self._rec
        import sherpa_onnx

        self._rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(self.model_dir / "model.int8.onnx"),
            tokens=str(self.model_dir / "tokens.txt"),
            num_threads=self.num_threads,
            use_itn=True,
            language="zh",
            debug=False,
        )
        return self._rec

    def transcribe(self, samples: np.ndarray, sr: int) -> dict:
        if not self.available():
            return {"text": "", "emotion": None, "event": None, "quality": 0.0, "error": "模型缺失"}
        import librosa

        audio = np.asarray(samples, dtype=np.float32)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        rec = self._ensure()
        with self._lock:
            stream = rec.create_stream()
            stream.accept_waveform(16000, audio)
            rec.decode_stream(stream)
            result = stream.result

        text = SPECIAL.sub("", getattr(result, "text", "") or "").strip()
        if not VALID.search(text):
            text = ""
        raw_emo = (getattr(result, "emotion", "") or "").strip("<>|").upper()
        event = (getattr(result, "event", "") or "").strip("<>|") or None
        duration = len(audio) / sr
        sentences = self._split_sentences(
            list(getattr(result, "tokens", []) or []),
            list(getattr(result, "timestamps", []) or []),
        )
        return {
            "text": text,
            "emotion": EMO_MAP.get(raw_emo),
            "event": event,
            "quality": speech_quality(text, duration),
            "duration": round(duration, 2),
            "sentences": sentences,
        }

    @staticmethod
    def _split_sentences(tokens: list[str], stamps: list[float]) -> list[dict]:
        """按标点切句，带时间戳。"""
        out: list[dict] = []
        buffer: list[str] = []
        start: float | None = None
        for token, ts in zip(tokens, stamps):
            clean = SPECIAL.sub("", token)
            if not clean:
                continue
            if start is None:
                start = ts
            buffer.append(clean)
            joined = "".join(buffer)
            if clean in SENT_END or (clean in SOFT and len(joined) >= 12):
                text = joined.strip()
                if VALID.search(text):
                    out.append({"t0": round(start, 2), "t1": round(ts, 2), "text": text})
                buffer, start = [], None
        if buffer and start is not None:
            text = "".join(buffer).strip()
            if VALID.search(text):
                last = stamps[-1] if stamps else start
                out.append({"t0": round(start, 2), "t1": round(last, 2), "text": text})
        return out
