"""声音事件（YAMNet，CPU，可选；模型 15.4 MB 不随包）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

EVENT_LABELS = {
    "Aircraft": "飞机", "Aircraft engine": "飞机引擎", "Jet engine": "喷气引擎",
    "Propeller, airscrew": "螺旋桨", "Helicopter": "直升机",
    "Fixed-wing aircraft, airplane": "固定翼飞机", "Vehicle": "车辆",
    "Motor vehicle (road)": "公路车辆", "Truck": "卡车", "Motorcycle": "摩托",
    "Engine": "引擎", "Light engine (high frequency)": "高频引擎",
    "Medium engine (mid frequency)": "中频引擎", "Heavy engine (low frequency)": "低频引擎",
    "Explosion": "爆炸", "Gunshot, gunfire": "枪声", "Machine gun": "机枪",
    "Artillery fire": "炮击", "Boom": "轰鸣", "Siren": "警报", "Alarm": "警铃",
    "Thunder": "雷声", "Wind": "风声", "Rain": "雨声", "Crowd": "人群",
    "Shout": "喊叫", "Screaming": "尖叫",
    # 社交/情绪类（ASR 转写里最有价值的部分）
    "Laughter": "笑声", "Giggle": "轻笑", "Snicker": "偷笑",
    "Belly laugh": "大笑", "Chuckle, chortle": "轻笑",
    "Applause": "掌声", "Cheering": "欢呼", "Cough": "咳嗽",
    "Sneeze": "打喷嚏", "Sigh": "叹气", "Crying, sobbing": "哭声",
    "Baby cry, infant cry": "婴儿哭声", "Whispering": "低语",
    "Burping, eructation": "打嗝", "Hiccup": "打嗝",
}

SOCIAL_TAGS = {
    "笑声", "轻笑", "偷笑", "大笑", "掌声", "欢呼", "咳嗽", "打喷嚏",
    "叹气", "哭声", "婴儿哭声", "低语", "打嗝",
}

MUSIC_TAGS = {
    "Music", "Pop music", "Hip hop music", "Rock music", "Soul music", "Funk",
    "Reggae", "Singing", "Rapping", "Classical music", "Electronic music",
    "House music", "Electronic dance music", "Ambient music", "Folk music",
    "Swing music", "Afrobeat", "Jingle, tinkle", "Musical instrument",
    "String section", "Sad music", "Music for children",
}
SPEECH_TAGS = {
    "Speech", "Narration, monologue", "Speech synthesizer",
    "Child speech, kid speaking", "Hubbub, speech noise, speech babble",
}
NOISE_TAGS = {
    "飞机", "飞机引擎", "喷气引擎", "螺旋桨", "直升机", "固定翼飞机", "车辆",
    "公路车辆", "卡车", "摩托", "引擎", "高频引擎", "中频引擎", "低频引擎",
    "爆炸", "枪声", "机枪", "炮击", "轰鸣", "警报", "警铃", "雷声", "风声", "雨声",
}


class SoundEvents:
    def __init__(self, model: Path | None, classes: Path | None):
        self.model = Path(model) if model else None
        self.classes = Path(classes) if classes else None
        self._session = None
        self._names: dict[int, str] = {}

    def available(self) -> bool:
        return bool(self.model and self.model.is_file() and self.classes and self.classes.is_file())

    def _ensure(self):
        if self._session is not None:
            return self._session
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(self.model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        for line in self.classes.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split(",", 2)
            if len(parts) == 3:
                self._names[int(parts[0])] = parts[2].strip().strip('"')
        return self._session

    def analyze(self, samples: np.ndarray, sr: int) -> dict:
        """返回 {music, speech, noise, events}（比例 0~1）。"""
        if not self.available():
            return {"ok": False}
        import librosa

        audio = np.asarray(samples, dtype=np.float32)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        try:
            session = self._ensure()
            _emb, scores = session.run(None, {"waveform": audio})
        except Exception:  # noqa: BLE001
            return {"ok": False}
        scores = np.asarray(scores, dtype=np.float32)
        index = {name: i for i, name in self._names.items()}
        frames = scores.shape[0]

        def mean_of(tags: set[str]) -> float:
            cols = [index[t] for t in tags if t in index]
            return float(scores[:, cols].max(axis=1).mean()) if cols else 0.0

        music = round(mean_of(MUSIC_TAGS), 3)
        speech = round(mean_of(SPEECH_TAGS), 3)
        hop = (len(audio) / 16000) / max(frames, 1)

        # 逐帧取「主导事件」，再按标签合并成时间段
        timeline: list[dict] = []
        for name, zh in EVENT_LABELS.items():
            if name not in index:
                continue
            col = scores[:, index[name]]
            threshold = 0.15 if zh in SOCIAL_TAGS else 0.3
            active = col >= threshold
            if not active.any():
                continue
            start = None
            for i, flag in enumerate(active):
                if flag and start is None:
                    start = i
                elif not flag and start is not None:
                    if (i - start) * hop >= 0.4:
                        timeline.append(
                            {
                                "t0": round(start * hop, 2),
                                "t1": round(i * hop, 2),
                                "label": zh,
                                "conf": round(float(col[start:i].max()), 3),
                            }
                        )
                    start = None
            if start is not None and (frames - start) * hop >= 0.4:
                timeline.append(
                    {
                        "t0": round(start * hop, 2),
                        "t1": round(frames * hop, 2),
                        "label": zh,
                        "conf": round(float(col[start:].max()), 3),
                    }
                )
        timeline.sort(key=lambda item: item["t0"])
        # 合并重叠的同类事件
        merged: list[dict] = []
        for item in timeline:
            if merged and merged[-1]["label"] == item["label"] and item["t0"] <= merged[-1]["t1"] + 0.3:
                merged[-1]["t1"] = max(merged[-1]["t1"], item["t1"])
                merged[-1]["conf"] = max(merged[-1]["conf"], item["conf"])
            else:
                merged.append(dict(item))
        timeline = merged[:40]

        events: dict[str, float] = {}
        for item in timeline:
            if item["label"] in NOISE_TAGS or item["label"] in SOCIAL_TAGS:
                events[item["label"]] = max(events.get(item["label"], 0.0), item["conf"])
        return {
            "ok": True,
            "music": music,
            "speech": speech,
            "noise": round(sum(events.values()) / max(len(events), 1), 3) if events else 0.0,
            "events": sorted(events.items(), key=lambda kv: -kv[1])[:6],
            "timeline": timeline,
            "frames": frames,
        }
