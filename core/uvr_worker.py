"""UVR 分离 worker：在独立进程里跑，避免和其它模型抢内存/分裂堆。

用法（由 separate.py 调起）：
    python uvr_worker.py <输入16k wav> <UVR 模型 onnx> <输出 vocals wav>
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as f:
        sr = f.getframerate()
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        if f.getnchannels() > 1:
            data = data.reshape(-1, f.getnchannels()).mean(axis=1).astype(np.int16)
    return data.astype(np.float32) / 32768.0, sr


def write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    data = np.clip(np.asarray(samples, dtype=np.float32), -1, 1)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((data * 32767).astype(np.int16).tobytes())


CHUNK = 10.0   # 每块秒数（越小越省内存）
OVERLAP = 0.5  # 块间重叠
MIN_CHUNK = 3.0


THREADS = 4


def separate_chunk(samples, sr, model, depth: int = 0):
    cfg = sherpa_onnx.OfflineSourceSeparationModelConfig(
        uvr=sherpa_onnx.OfflineSourceSeparationUvrModelConfig(model=str(model)),
        num_threads=THREADS,
        debug=False,
    )
    sep = sherpa_onnx.OfflineSourceSeparation(
        sherpa_onnx.OfflineSourceSeparationConfig(model=cfg)
    )
    try:
        result = sep.process(sr, np.stack([samples, samples]))
    except Exception as exc:  # noqa: BLE001
        seconds = len(samples) / sr
        if depth >= 3 or seconds <= MIN_CHUNK:
            raise
        print(f"chunk {seconds:.1f}s 分配失败，二分重试：{str(exc)[:60]}", file=sys.stderr)
        half = len(samples) // 2
        left = separate_chunk(samples[:half], sr, model, depth + 1)
        right = separate_chunk(samples[half:], sr, model, depth + 1)
        return np.concatenate([left, right])
    vocals = np.asarray(result.stems[0].data, dtype=np.float32)
    if vocals.ndim > 1:
        vocals = vocals.mean(axis=0)
    vocals = vocals.reshape(-1)
    if result.sample_rate != 16000:
        import librosa

        vocals = librosa.resample(vocals, orig_sr=result.sample_rate, target_sr=16000)
    return vocals


def main() -> int:
    if len(sys.argv) < 4:
        print("usage: uvr_worker.py <in.wav> <model.onnx> <out.wav>", file=sys.stderr)
        return 2
    src, model, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    global THREADS
    if len(sys.argv) > 4:
        try:
            THREADS = max(1, int(sys.argv[4]))
        except ValueError:
            pass
    samples, sr = read_wav(src)

    chunk = int(CHUNK * sr)
    step = int((CHUNK - OVERLAP) * sr)
    pieces: list[np.ndarray] = []
    pos = 0
    while pos < len(samples):
        block = samples[pos: pos + chunk]
        if len(block) < sr // 2:
            break
        pieces.append(separate_chunk(block, sr, model))
        pos += step

    if not pieces:
        print("no audio", file=sys.stderr)
        return 3

    # 按块步长拼接（重叠部分取后一块，简单可靠）
    total = int(step * (len(pieces) - 1) + len(pieces[-1]))
    out = np.zeros(total, dtype=np.float32)
    for index, piece in enumerate(pieces):
        offset = index * step
        end = min(offset + len(piece), total)
        out[offset:end] = piece[: end - offset]

    write_wav(dst, out, 16000)
    print(f"ok chunks={len(pieces)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
