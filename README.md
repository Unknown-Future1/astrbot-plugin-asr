
# 🐋 Sensevoice ASR语音转写增强 — AstrBot 插件

<p align="center">
  把 QQ 语音「听懂」再交给大模型：<strong>VAD 人声检测 + 音乐分离 + 声音事件 + SenseVoice 转写</strong>。<br/>
  纯 CPU / ONNX 推理，<strong>不需要 torch、不需要显卡、不依赖任何云端 API</strong>。
</p>
<p align="center">
  <img src="https://img.shields.io/badge/version-0.2.0-blue" alt="version" />
  <img src="https://img.shields.io/badge/AstrBot-%E2%89%A5%204.29.4-orange" alt="astrbot" />
  <img src="https://img.shields.io/badge/backend-SenseVoice%20%2B%20UVR%20%2B%20YAMNet-4f9cf9" alt="backend" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="license" />
</p>

---

## 本插件会自动把聊天中的语音进行文字转写

它做的是 **STT Provider**：把语音转成文字后，作为「用户说了什么」喂给后台大模型，
再由模型用自己的口吻回复。

```
QQ 语音 → 插件 → 文字（含笑声/掌声等标注） → 大模型 → 群回复
```

---

## ✨ 功能特性

| 特性 | 说明 |
| --- | --- |
| 🎙️ **本地识别** | SenseVoice-Small（int8，228 MB），中英日韩粤，CPU 推理 |
| 🧹 **音乐分离** | 人声与 BGM 混合时自动 UVR 分离，分离后识别质量明显提升 |
| 👂 **VAD 人声检测** | 纯音乐/静音直接返回「没有人声」，不瞎编文字 |
| 🔔 **声音事件** | YAMNet 识别笑声/掌声/咳嗽等，按时间戳内联进文本 |
| ⏱️ **事件时间戳** | 例如「你们在说什么啊（笑声 3.2s）我也想听听」 |
| 🎭 **情绪前缀** | 识别出情绪时加「（开心）」前缀（默认开启，可关） |
| 🛟 **低质量兜底** | 转写不可信时再尝试分离抢救，仍不行返回「没听清」提示，避免垃圾文本 |
| 🧩 **自动注册 STT** | 首次启动自动注册 provider `asr-enhance-local` 并设为默认 |
| 🔧 **ffmpeg 自愈** | 自动把 ffmpeg 复制到数据目录，副本被清理后自动重建 |
| 🧵 **线程自适应** | `num_threads=0` 时按 CPU 核心数取 2~6，兼顾速度与占用 |
| 🛡️ **优雅降级** | 模型缺失/识别失败只记日志，不吞掉正常对话 |
| 💤 **无常驻服务** | 没有后台 daemon，不占端口，不占显存 |

---

## 📋 前置要求

1. **AstrBot** ≥ v4.29.4 已安装并正常运行

2. **模型组件包** 让DSH或豆包等大模型代劳。下载并解压到任意目录（如`H:\ASR_Enhance_Scripts`中）

   ![image-20261127182311161](C:\Users\Unknown_Future\AppData\Roaming\Typora\typora-user-images\image-20261127182311161.png)

   - 仓库：<https://github.com/Unknown-Future1/ASR_Enhance_Scripts>

3. 无需显卡、无需 torch、无需安装 ffmpeg 到系统 PATH（模型包里自带）

### 模型包结构

> 组件包只从上面这个 Release 下载；插件本体不附带模型。

> 组件包只从上面这个 Release 下载；插件本体与本仓库都不附带模型。

```
ASR_Enhance_Scripts\
├─ sensevoice\          ← SenseVoice 模型（model.int8.onnx + tokens.txt，228 MB）
├─ yamnet\              ← 声音事件模型（yamnet.onnx + 类别表，15.4 MB）
├─ uvr\                 ← 人声分离模型（UVR-MDX-NET-Voc_FT.onnx，63.7 MB）
├─ ffmpeg\              ← ffmpeg.exe（61.5 MB，目录层级随意）
└─ README.txt
```

> 插件会自动在包内找子目录；`ffmpeg.exe` 放在包根、`ffmpeg\`、`ffmpeg\bin\` 都能识别。

---

## 🚀 初步设置（装完插件 3 步）

1. **装插件**：AstrBot 面板 → 插件市场搜索「Sensevoice ASR语音转写增强」（或 `astrbot_plugin_sensevoice_asr_enhance`），安装

2. **下载模型包**：从 [Releases](https://github.com/Unknown-Future1/ASR_Enhance_Scripts/releases) 下载 `ASR_Enhance_Scripts.zip`，解压到任意目录，例如 `D:\ASR_Enhance_Scripts`

3. **填模型目录**：插件设置里「本地模型包目录」填这个文件夹，保存 → **重启 AstrBot**

首次启动会自动：

- 注册 STT provider `asr-enhance-local`
- 设为默认语音转文字 provider
- 把 ffmpeg 复制到 `data/plugin_data/astrbot_plugin_sensevoice_asr_enhance/bin/`

> 第一次启动日志提示「已注册…重启后生效」属正常，再重启一次即可。
> 之后日志会显示 `[ASR+] STT provider 就绪 | SenseVoice=… | 分离=UVR | YAMNet=True | ffmpeg=…`

---

## 🔄 处理流程

```
QQ 语音
  │
  ├─ 转码：任意格式 → 16k 单声道 wav（ffmpeg，副本自愈）
  │
  ├─ 分析
  │    ├─ VAD：人声占比 / 段数
  │    └─ YAMNet：音乐占比 / 语音占比 / 事件时间线
  │
  ├─ 判型
  │    ├─ 人声 < 2%              → 返回「[这段语音里没有人声]」
  │    ├─ 音乐 ≥50% 且 语音 <50% → UVR 分离（10 秒分块，失败自动二分重试）
  │    └─ 其余                    → 原样识别
  │
  ├─ SenseVoice 识别（带句子时间戳）
  │
  ├─ 质量 < 0.5？
  │    ├─ 有音乐且没分离过 → 分离抢救 → 再识别
  │    └─ 仍低 → 返回「没听清」提示
  │
  ├─ 事件标注（社交类全标；置信度低加「疑似」）
  │
  └─ 情绪前缀（可关）→ 返回文字给大模型
```

### 判型规则

| 条件 | 处理 | 理由 |
| --- | --- | --- |
| 人声占比 < 2% | 不做识别，返回无人声提示 | 纯音乐/静音，硬识别只会出乱码 |
| 音乐 ≥50% 且 语音 <50% | UVR 分离后再识别 | 人声与 BGM 混合，ASR 直接听会很差 |
| 其余 | 原样识别 | 人声为主，分离反而浪费时间 |

### 输出示例

| 场景 | 返回给大模型的内容 |
| --- | --- |
| 正常语音 | `离谱男子工作近9年，离职后道歉公司17万元…` |
| 有笑声 | `你们在说什么啊（笑声 3.2s）我也想听听` |
| 低置信事件 | `…（疑似掌声 5.0s）` |
| 纯音乐/静音 | `[这段语音里没有人声]` |
| 识别不可信 | `[这段语音没听清（可能是方言、口音，或背景音太重），可以卖萌表示没听懂]` |
| 开情绪 | `（开心）你们在说什么啊（笑声 3.2s）我也想听听` |

---

## 📊 实测耗时（CPU）

| 场景 | 耗时 |
| --- | --- |
| 短语音 <10s | 0.3 ~ 0.6s |
| 20s 干净语音 | 0.8s |
| 64s 新闻 | 4.5s |
| 20s 混合音频（分离） | 13 ~ 16s |
| 44s 混合音频（分离） | 45 ~ 47s |
| 静音 | 0.03s |



---

## ⚙️ 配置说明

安装后在 **插件配置 → Sensevoice ASR语音转写增强** 中修改。

### 基础开关

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `enable_sensevoice` | `bool` | `true` | 总开关：关闭后无法给后台输入语音文字 |
| `model_root` | `string` | `H:\ASR_Enhance_Scripts` | 模型包目录，插件自动找包内组件 |
| `enable_vad` | `bool` | `true` | 人声检测，防止纯音乐被硬转成文字 |
| `enable_separate` | `bool` | `true` | 音乐主导时自动分离人声 |
| `enable_yamnet` | `bool` | `true` | 声音事件识别（笑声/掌声/咳嗽…） |

### 输出控制

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `annotate_events` | `bool` | `true` | 在文本里标注事件：确定写「（笑声 5.2s）」，低置信写「（疑似掌声 5.0s）」 |
| `with_emotion` | `bool` | `true` | 给文本加「（开心）」这类情绪前缀 |
| `reply_with_audio` | `bool` | `true` | 文本后附一行「音乐占比 / 人声占比 / 音效」（chat 模式） |
| `trigger_mode` | `string` | `always` | `always`=所有语音；`at`=@机器人才处理；`reply`=只处理引用语音 |
| `mode` | `string` | `provider` | `provider`=喂给大模型（推荐）；`chat`=插件直接把文字发群里 |

### 兜底与限制

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `quality_threshold` | `float` | `0.5` | 转写可信度低于此值 → 分离抢救 → 仍低则返回提示；越低越宽松 |
| `dialect_reply` | `string` | `[这段语音没听清…]` | 识别不可信时返回给模型的内容 |
| `silent_reply` | `string` | `[这段语音里没有人声]` | 纯音乐/静音时返回的内容 |
| `separate_max_seconds` | `int` | `0` | 自动分离的时长上限（秒），`0`=不限制 |
| `max_seconds` | `int` | `180` | 单条语音最长处理秒数（QQ 上限 3 分钟） |
| `num_threads` | `int` | `0` | 推理线程数，`0`=按 CPU 核心数取 2~6；作用于 SenseVoice / UVR |

---

## 🧠 关于质量分

```
q = 语速分 × 用字多样性分 × 功能词分
  语速：1.5 ~ 8 字/秒 → 1.0，否则 0.25
  多样性：不同汉字占比 ≥ 0.42 → 1.0
  功能词：「的/了/是/我/你…」占比 ≥ 0.045 → 1.0
```

明显错乱（如 44 秒只出 29 个字）会掉到 0.25，触发兜底或分离抢救。

---

## 🔧 常见问题排查

### 发了语音，模型还是没反应

按顺序检查：

1. 日志里是否有 `[ASR+] STT provider 就绪`
2. AstrBot 的语音转文字 provider 是否选中 `asr-enhance-local`
3. 「本地模型包目录」是否填对、包内是否有 `sensevoice\model.int8.onnx`
4. 是否 `mode=provider`（`chat` 模式不会喂给模型）

### 日志报 ffmpeg 相关权限错误

插件会把 ffmpeg 复制到数据目录再执行（`data/plugin_data/astrbot_plugin_sensevoice_asr_enhance/bin/ffmpeg.exe`）。
如果副本被清理，下次使用会自动重建。若持续失败，检查数据目录是否可写。

### 纯音乐被识别出奇怪的文字

确认 `enable_vad=true`。VAD 会把人声占比 <2% 的音频直接判为「没有人声」。

### 有 BGM 的语音识别很差

确认 `enable_separate=true`。混合音频会走 UVR 分离，分离后识别质量会明显提升，但耗时增加（约 0.55x 实时）。

### 短暂语音被判定为「没听清」

`quality_threshold` 默认 0.5 偏严，可降到 0.2~0.3；或者检查该语音是否真的只有一两个词。

### 识别慢

- 看日志是 `none` 还是 `separate`：分离本来就慢
- `num_threads` 保持 0（自动）或固定 4；不要设 1
- 3 分钟混合音频约需 100 秒，属正常

### 事件标注太多/太少

- 标注只包含社交类（笑声/掌声/咳嗽/叹气/低语…），噪声类永远不会进文本
- 置信度 <0.35 的事件会带「疑似」前缀
- 不想要标注就把 `annotate_events` 关掉

---

## 🔗 相关链接

- 🧩 [模型组件包（本插件必需）](https://github.com/Unknown-Future1/ASR_Enhance_Scripts)
- 📘 [AstrBot 插件开发文档](https://docs.astrbot.app/dev/star/plugin-new.html)
- 🎙️ [SenseVoice（FunAudioLLM）](https://github.com/FunAudioLLM/SenseVoice)
- 🧠 [sherpa-onnx（推理框架）](https://github.com/k2-fsa/sherpa-onnx)
- 🎵 [UVR-MDX 人声分离模型](https://github.com/Anjok07/ultimatevocalremovergui)

---

## 📄 许可证

本插件代码采用 [MIT License](LICENSE) 开源。

> **注意**：MIT 只覆盖本插件的代码。
> SenseVoice / YAMNet / UVR 模型各有自己的许可证，使用时请遵守对应模型仓库的条款。
