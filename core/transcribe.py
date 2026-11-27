"""SenseVoice 识别（供增强插件自检用；模型外置）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_NAME = "sensevoice-local"


class Transcriber:
    def __init__(self, model_dir: Path | None, num_threads: int = 4):
        self.model_dir = Path(model_dir) if model_dir else None
        self.num_threads = num_threads
        self._rec = None

    def available(self) -> bool:
        return bool(self.model_dir and (self.model_dir / "model.int8.onnx").is_file())

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

    def transcribe(self, samples: np.ndarray, sr: int) -> str:
        if not self.available():
            return ""
        import librosa

        audio = np.asarray(samples, dtype=np.float32)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        rec = self._ensure()
        stream = rec.create_stream()
        stream.accept_waveform(16000, audio)
        rec.decode_stream(stream)
        text = (stream.result.text or "").strip()
        return text
