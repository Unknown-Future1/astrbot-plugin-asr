"""UVR-MDX 人声分离（onnxruntime / sherpa-onnx，纯 CPU，不需要 torch）。"""

from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np


def _write_wav(path: Path, samples: np.ndarray, sr: int) -> Path:
    data = np.clip(np.asarray(samples, dtype=np.float32), -1, 1)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((data * 32767).astype(np.int16).tobytes())
    return path


WORKER = Path(__file__).resolve().parent / "uvr_worker.py"


def _write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    data = np.clip(np.asarray(samples, dtype=np.float32), -1, 1)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((data * 32767).astype(np.int16).tobytes())


class Separator:
    def __init__(self, uvr_model: Path | None = None, num_threads: int = 4):
        self.uvr_model = Path(uvr_model) if uvr_model else None
        self.num_threads = num_threads

    def available(self) -> bool:
        return bool(self.uvr_model and self.uvr_model.is_file())

    def separate(self, samples: np.ndarray, sr: int, out_dir: Path) -> dict:
        if not self.available():
            return {"ok": False, "error": "未配置 UVR 模型"}
        out_dir.mkdir(parents=True, exist_ok=True)
        sub = self._run_subprocess(samples, sr, out_dir)
        if sub.get("ok"):
            return sub
        sub2 = self._run_subprocess(samples, sr, out_dir)
        if sub2.get("ok"):
            return sub2
        first = self._run(samples, sr, out_dir)
        if first.get("ok"):
            return first
        # 部分环境第一次会因 onnxruntime 内存分配失败，清理后重试一次
        try:
            import gc

            gc.collect()
        except Exception:  # noqa: BLE001
            pass
        second = self._run(samples, sr, out_dir)
        if second.get("ok"):
            return second
        second["error"] = (first.get("error") or "")[:120] + " / retry: " + (second.get("error") or "")[:120]
        return second

    def _run_subprocess(self, samples: np.ndarray, sr: int, out_dir: Path) -> dict:
        """独立进程跑 UVR：避免和 SenseVoice/YAMNet 抢内存导致分配失败。"""
        import subprocess
        import sys

        if not WORKER.is_file():
            return {"ok": False, "error": "缺少 uvr_worker.py"}
        started = time.time()
        try:
            src = out_dir / "uvr_input.wav"
            dst = out_dir / "vocals.wav"
            if dst.exists():
                dst.unlink()
            _write_wav(src, samples, sr)
            proc = subprocess.run(
                [
                    sys.executable, str(WORKER), str(src), str(self.uvr_model),
                    str(dst), str(self.num_threads),
                ],
                capture_output=True,
                timeout=600,
            )
            if dst.is_file() and dst.stat().st_size > 1000 and dst.stat().st_mtime >= started - 1:
                return {
                    "ok": True,
                    "wav": dst,
                    "method": "uvr",
                    "seconds": round(time.time() - started, 2),
                }
            return {
                "ok": False,
                "error": (proc.stderr or proc.stdout or b"")[-160:].decode("utf-8", "ignore"),
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)[:160]}

    def _run(self, samples: np.ndarray, sr: int, out_dir: Path) -> dict:
        try:
            import librosa
            import sherpa_onnx

            cfg = sherpa_onnx.OfflineSourceSeparationModelConfig(
                uvr=sherpa_onnx.OfflineSourceSeparationUvrModelConfig(
                    model=str(self.uvr_model)
                ),
                num_threads=self.num_threads,
                debug=False,
            )
            sep = sherpa_onnx.OfflineSourceSeparation(
                sherpa_onnx.OfflineSourceSeparationConfig(model=cfg)
            )
            started = time.time()
            out = sep.process(sr, np.stack([samples, samples]))
            vocals = np.asarray(out.stems[0].data, dtype=np.float32)
            if vocals.ndim > 1:
                vocals = vocals.mean(axis=0)
            vocals16 = librosa.resample(
                vocals.reshape(-1), orig_sr=out.sample_rate, target_sr=16000
            )
            path = _write_wav(out_dir / "vocals.wav", vocals16, 16000)
            return {
                "ok": True,
                "wav": path,
                "method": "uvr",
                "seconds": round(time.time() - started, 2),
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)[:160]}
