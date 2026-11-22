# ⚠️ 半成品（WIP）

> 基础功能已实现，仍在修改完善；接口、参数可能随时变动，请勿用于生产。

---

# astrbot_plugin_sensevoice_asr

本地 SenseVoice 语音识别插件：**QQ 语音 → 文字**，可选情绪标注，零 API Key、零联网。

## 一、它做什么

向 AstrBot 注册一个 STT provider（类型 `sensevoice_sherpa_stt`）。默认使用本地 SenseVoice；如果用户切换为其他 STT provider，插件会接收语音预处理后写入消息的转写文本，供 `/asr` 复用：

- 群友发语音 → 自动转成文字再交给 LLM
- 转写结果可带情绪前缀：`（开心）今天好热`
- 纯音乐 / 静音语音 → 返回空字符串，不会产生"盲回"

## 二、它不做什么

| 不做 | 原因 |
| --- | --- |
| 硬字幕 OCR | 属于视频理解插件 |
| 软字幕解析 | 只有视频文件有字幕轨，语音消息用不上 |
| 实时流式转写 | 当前是整段转写，60 秒语音约 0.5 秒 |

## 三、安装

### 目录结构

```
astrbot_plugin_sensevoice_asr/
├── main.py                 # 插件入口（/asr、/asr_status 指令）
├── metadata.yaml
├── _conf_schema.json       # 面板配置
├── requirements.txt        # sherpa-onnx / numpy / opencc
├── core/
│   ├── provider.py         # STT provider（注册入口）
│   ├── recognizer.py       # SenseVoice 封装
│   ├── qwen3.py            # Qwen3-ASR 0.6B int8 封装
│   ├── transcription.py    # 级联策略与跨插件共享接口
│   └── audio.py            # ffmpeg → 16k 单声道 PCM
└── models/sensevoice/      # model.int8.onnx + tokens.txt（228 MB）
```

### 安装步骤

1. 把整个目录放到 `AstrBot/data/plugins/`
2. 面板 → 插件管理 → 安装并启用插件，确认依赖（sherpa-onnx、opencc）装好
3. 插件首次加载时会自动添加本地 SenseVoice STT provider，并设为默认语音转文字 provider
4. 重启 AstrBot 后，发一条语音即可由本地模型转写并交给 LLM

自动设置只在首次启用时执行一次；之后你可以在面板切换默认 STT，插件不会在每次启动时改回。切换到其他 provider 后，语音识别与送入 LLM 由该 provider 负责；本插件会缓存 QQ 入站语音的预处理转写，供 `/asr` 查询。

## 四、配置项

| 项 | 默认 | 说明 |
| --- | --- | --- |
| `model_dir` | 空 | 留空自动用插件内的 `models/sensevoice` |
| `with_emotion` | true | 是否在文字前加「（情绪）」 |
| `num_threads` | 6 | 推理线程数 |
| `max_seconds` | 180 | 单条语音上限，超过拒绝 |
| `ffmpeg_path` | 空 | 留空用 PATH 里的 ffmpeg |
| `keep_loaded` | true | 模型常驻（约 330 MB 内存），关闭则每次重载（+1.4s） |
| `qwen3_enabled` | true | 启用 Qwen3-ASR 低可信兜底；模型按需加载 |
| `qwen3_model_dir` | 自动查找 | Qwen3-ASR 模型目录，包含三个 ONNX 文件和 `tokenizer/` |
| `qwen3_num_threads` | 2 | Qwen3 CPU 推理线程数 |
| `qwen3_max_new_tokens` | 256 | Qwen3 单次生成 token 上限 |
| `qwen3_keep_loaded` | true | Qwen3 首次兜底后常驻内存；关闭则每次兜底重新加载 |

Qwen3 模型路径按以下顺序查找：`qwen3_model_dir` 配置、环境变量 `ASTRBOT_SENSEVOICE_QWEN3_DIR`、`data/plugin_data/astrbot_plugin_sensevoice_asr/models/qwen3-asr-0.6b`。开发机也会检查 `H:\Deepseek-Harness\Workspace\asr\models\qwen3-asr-0.6b`。找不到或模型加载失败时，插件保留 SenseVoice 结果，不会让 STT 整体报错。

默认级联策略是 SenseVoice 先识别；文本质量分低于 `quality_threshold` 时再调用 Qwen3。该分数只是启发式触发器，后续需要用实际 QQ 语音和视频音轨测试后调整。

### 给视频理解插件复用

视频插件可以通过 AstrBot context 取得已经加载的同一服务，避免重复加载模型。服务输入为 16 kHz 单声道 `float32` PCM；视频侧负责先抽取音轨并做 VAD/分段：

```python
from astrbot_plugin_sensevoice_asr.core.transcription import (
    get_transcription_service,
)

service = get_transcription_service(self.context)
# mode="cascade": SenseVoice 优先，低质量时走 Qwen3
# mode="qwen3": 直接用 Qwen3 识别该段音频
result = await asyncio.get_running_loop().run_in_executor(
    None, lambda: service.recognize_samples(audio_samples, mode="cascade")
)
text = result["text"]
```

返回结果包含 `engine`、`primary_text`、`fallback_used` 和 `fallback_reason`，便于视频插件记录每段音频由哪个引擎处理。长视频应先切成语音片段再调用，避免单次生成长度过大。

## 五、指令

| 指令 | 作用 |
| --- | --- |
| `/asr`（别名 `/听一下`、`/转写`） | 转写你引用的那条语音消息 |
| `/asr_status` | 查看模型路径与 provider 状态 |

## 六、实测数据

| 用例 | 结果 |
| --- | --- |
| 10 秒普通话语音 | `（平静）桌上放着三个苹果，老王面馆的营业时间是早上6点半到晚上9点，人均28元。` |
| 纯音乐 5 秒 | `""`（正确判为空） |
| 加载耗时 | 1.4 秒（常驻后不再加载） |
| 转写耗时 | 10 秒音频约 0.2 秒 |
| 内存占用 | 约 330 MB |

## 七、已知限制

| 限制 | 说明 |
| --- | --- |
| 方言 | 仅普通话 + 粤语；四川话/东北话等会错 |
| 多人同时说话 | 会混在一起，无说话人分离 |
| 低码率音频 | 39 kbps 以下准确率明显下降 |
| 情绪粒度 | 整段一个情绪（语音消息通常很短，可接受） |

## 八、与视频理解插件的关系

两者共用同一套 SenseVoice 模型与识别逻辑，但**各自加载一份**（各 330 MB）。

如果希望共享，可后续把识别器抽成独立本地服务，两边都通过 HTTP 调用。

## 九、方言检测（低质量转写拦截）

SenseVoice 只支持普通话 + 粤语；遇到其他方言会「按普通话硬解」，输出看似正常但错误的文字。
插件用三个轻量指标打分，低于阈值就**不把错误文字交给模型**，而是返回一句提示。

### 判定指标

| 指标 | 正常范围 | 说明 |
| --- | --- | --- |
| 语速 | 1.5~8 字/秒 | 方言硬解常出现 0.5 字/秒（只蹦几个字）或 9+ 字/秒（乱码堆字） |
| 字符多样性 | ≥ 0.42 | 重复字多说明在硬凑 |
| 功能词比例 | ≥ 0.06 | 正常中文必然有「的了是我你不在」等 |

三项相乘得到 `quality`（0~1），低于 `quality_threshold`（默认 0.5）即拦截。

### 拦截后的返回

```json
"dialect_reply": "[对方说了一段方言或口音很重的话，你没听懂，可以卖萌表示听不懂]"
```

这段文字会作为「语音转写结果」进入对话，模型看到后会自然地回一句听不懂。

### 实测

| 音频 | quality | 结果 |
| --- | --- | --- |
| 清晰普通话（TTS） | 1.0 | 正常转写 |
| 抖音旁白 | 1.0 | 正常转写 |
| 多人喊叫 / 极差音频 | **0.25** | 拦截，返回听不懂提示 |
| 纯音乐 | 0.0 | 返回空（不回复） |

## 十一、模型外置与服务启停（v0.8.0）

插件包**不含模型文件**（当前约 0.04 MB），模型放在外部按路径加载。

| 模型 | 需要的文件 | 默认路径 |
| --- | --- | --- |
| SenseVoice | `model.int8.onnx`、`tokens.txt` | `data/plugin_data/astrbot_plugin_sensevoice_asr/models/sensevoice` |

查找顺序：配置 `model_dir` → 环境变量 `ASTRBOT_SENSEVOICE_DIR` → 插件数据目录。

| 指令 | 作用 |
| --- | --- |
| `/asr_status` | 路径、文件完整性、是否已加载 |
| `/asr_start` | 按路径加载模型 |
| `/asr_stop` | 卸载模型，释放内存 |

> Qwen3-ASR 0.6B 兜底已在 v0.8.0 移除：实测对粤语/口音的增益有限，
> 长音频还需分段且速度慢，性价比不足。

