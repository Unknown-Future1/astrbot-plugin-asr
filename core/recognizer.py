"""SenseVoice 识别器封装：sherpa-onnx 推理 + 清理 + 可选情绪。"""

from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np

SPECIAL = re.compile(r"<\|[^|]*\|>")
VALID = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
PHANTOM_TAIL = re.compile(r"[，,。\s]*抖音[。\s]*$")

EMO_MAP = {
    "HAPPY": "开心",
    "SAD": "悲伤",
    "ANGRY": "生气",
    "NEUTRAL": "平静",
    "FEARFUL": "害怕",
    "DISGUSTED": "厌恶",
    "SURPRISED": "惊讶",
    "EMO_UNKNOWN": "未知",
    "OTHER": "其他",
}

# 常见中文功能词：正常句子几乎必然出现
FUNC_WORDS = re.compile(r"[的了是我你他她它不在有这那就都也还和吧呢啊嘛哦么什怎吗]")
CJK = re.compile(r"[\u4e00-\u9fff]")


def speech_quality(text: str, duration: float) -> tuple[float, str]:
    """粗判转写是否可信，返回 (分数 0~1, 原因)。

    针对方言/严重失真音频：SenseVoice 会「按普通话硬解」，输出语速异常、
    重复率高、缺少功能词的文本。这里用三个轻量指标兜底。
    """
    chars = CJK.findall(text)
    n = len(chars)
    if n == 0:
        return 0.0, "empty"
    cps = n / max(duration, 0.3)
    speed = 1.0 if 1.5 <= cps <= 8.0 else 0.25
    uniq = len(set(chars)) / n
    diversity = 1.0 if uniq >= 0.42 else 0.3
    func = len(FUNC_WORDS.findall(text))
    func_ratio = func / n
    function_score = 1.0 if func_ratio >= 0.06 else 0.35
    score = round(speed * diversity * function_score, 3)
    reason = f"speed={cps:.1f}cps uniq={uniq:.2f} func={func_ratio:.2f}"
    return score, reason


try:  # 可选依赖，没装也能用
    import opencc

    _CC = opencc.OpenCC("t2s")
except Exception:  # noqa: BLE001
    _CC = None


def to_simplified(text: str) -> str:
    if _CC is None or not text:
        return text
    try:
        return _CC.convert(text)
    except Exception:  # noqa: BLE001
        return text


class SenseVoiceRecognizer:
    """一个进程内只加载一次；内部加锁保证并发安全。"""

    def __init__(
        self,
        model_dir: Path,
        num_threads: int = 6,
        keep_loaded: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.num_threads = num_threads
        self.keep_loaded = keep_loaded
        self._rec = None
        self._lock = threading.Lock()

    @property
    def model_path(self) -> Path:
        return self.model_dir / "model.int8.onnx"

    @property
    def tokens_path(self) -> Path:
        return self.model_dir / "tokens.txt"

    def available(self) -> tuple[bool, str]:
        if not self.model_path.is_file():
            return False, f"缺少模型文件: {self.model_path}"
        if not self.tokens_path.is_file():
            return False, f"缺少 tokens: {self.tokens_path}"
        return True, ""

    def unload(self) -> bool:
        """释放模型（服务停止时调用），返回是否真的卸载了。"""
        with self._lock:
            had = self._rec is not None
            self._rec = None
        if had:
            import gc

            gc.collect()
        return had

    def loaded(self) -> bool:
        return self._rec is not None

    def _ensure(self):
        if self._rec is not None:
            return self._rec
        import sherpa_onnx

        ok, err = self.available()
        if not ok:
            raise RuntimeError(err)
        rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(self.model_path),
            tokens=str(self.tokens_path),
            num_threads=self.num_threads,
            use_itn=True,
            language="zh",
            debug=False,
        )
        if self.keep_loaded:
            self._rec = rec
        return rec

    def recognize(self, samples: np.ndarray) -> dict:
        """samples: float32 单声道 16kHz，返回 {text, emo, emo_raw, event}。"""
        if samples.size < 1600:
            return {"text": "", "emo": None, "emo_raw": None, "event": None}

        rec = self._ensure()
        with self._lock:
            stream = rec.create_stream()
            stream.accept_waveform(16000, samples)
            rec.decode_stream(stream)
            result = stream.result

        text = SPECIAL.sub("", getattr(result, "text", "") or "").strip()
        text = PHANTOM_TAIL.sub("", text)
        if not VALID.search(text):      # 纯标点/空白 = 没人声
            text = ""
        raw = (getattr(result, "emotion", "") or "").strip("<>|").upper()
        event = (getattr(result, "event", "") or "").strip("<>|")
        lang = (getattr(result, "lang", "") or "").strip("<>|").lower()
        duration = samples.size / 16000
        score, reason = speech_quality(text, duration)
        return {
            "text": to_simplified(text),
            "emo": EMO_MAP.get(raw) if raw else None,
            "emo_raw": raw or None,
            "event": event or None,
            "lang": lang or None,
            "quality": score,
            "quality_reason": reason,
            "duration": round(duration, 2),
        }
