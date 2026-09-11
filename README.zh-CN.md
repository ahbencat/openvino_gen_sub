# OpenVINO 视频字幕生成工具

[English](README.md) | **简体中文**

基于 OpenVINO 的端到端视频字幕生成与翻译工具，面向 Intel 平台。

- **转录** —— Whisper Large-v3 FP16 将视频/音频转为 SRT，配合 Silero VAD 按自然停顿分段，而非固定切片
- **翻译** —— Hy-MT2 / Hunyuan Dense V1 将 SRT 翻译为任意目标语言，时间轴原样保留
- **转换** —— 自行将 Hunyuan 模型导出为 OpenVINO FP16 或 INT4

开发与实测环境：Intel Core Ultra 7 265K + Arc B580。

## 流水线

```
video.mkv
    |  ffmpeg  ->  16 kHz 单声道 wav
    v
Silero VAD  ->  语音段
    |
    |  Whisper Large-v3 FP16 (OpenVINO, Arc GPU)
    v
source.srt
    |
    |  Hy-MT2 / Hunyuan Dense V1 (OpenVINO)
    v
translated.srt
```

Whisper 自带的 `translate` 任务只能输出英文。目标语言只要不是英文，就必须先转录、再翻译 —— 这正是两阶段流水线存在的原因。

## 组成部分

| 路径 | 作用 |
|------|------|
| `generate_and_translate.py` | 一键入口：视频/音频 → 原语言 SRT → 翻译后 SRT |
| `batch_generate.py` | 批量入口：扫描整个文件夹，处理其中每个视频 |
| `video_to_subtitle.py` | 视频/音频 → SRT（Whisper Large-v3 + Silero VAD） |
| `translate_srt.py` | SRT → 翻译后 SRT（Hy-MT2），时间轴不变 |
| `tools/model_conversion/` | Hunyuan → OpenVINO 导出（FP16 / INT4）+ 推理验证 |
| `models/` | 模型卡；权重下载到对应文件夹 |

## 环境要求

Python 3.10；转录环节需要 [ffmpeg](https://ffmpeg.org/)（加入 `PATH`，或用 `FFMPEG_PATH` 环境变量指向可执行文件）。

```bash
pip install -r requirements.txt
```

或手动组装环境：

```bash
conda create -n py310_openvino python=3.10
conda activate py310_openvino

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install openvino nncf optimum[openvino] transformers soundfile silero-vad
```

`nncf` 仅 INT4 量化需要 —— 只下载已转换模型的话可以跳过。

## 快速开始

### 1. 准备模型

两个模型都需要；权重不随仓库分发 —— 下载或自行转换到 `models/` 对应文件夹，脚本默认路径即可命中：

**Whisper Large-v3 FP16 (OpenVINO)** —— 从 [OpenVINO/whisper-large-v3-fp16-ov](https://huggingface.co/OpenVINO/whisper-large-v3-fp16-ov) 获取（Apache-2.0）：

```bash
hf download OpenVINO/whisper-large-v3-fp16-ov --local-dir models/whisper-large-v3-fp16-ov
```

**Hy-MT2** —— 下载已转换的 INT4 7B（ModelScope）：**[ahbencat/Hy-MT2-7B-ov-int4](https://modelscope.cn/models/ahbencat/Hy-MT2-7B-ov-int4)**（约 4.2 GB）：

```bash
modelscope download --model ahbencat/Hy-MT2-7B-ov-int4 --local_dir models/Hy-MT2-7B-ov-int4
```

或自行转换（1.8B FP16，或自定义 INT4 参数）—— 见 [`tools/model_conversion/README.zh-CN.md`](tools/model_conversion/README.zh-CN.md)。

模型放别处也行，用 `--model-dir` 指向即可。

### 2. 生成字幕

```bash
# 单个文件：转录 + 翻译（默认目标语言为中文）
python generate_and_translate.py "input.mkv"

# 指定源语言，输出英文
python generate_and_translate.py "input.mkv" --src-lang ja --tgt-lang English

# 只转录，不翻译
python generate_and_translate.py "input.mkv" --tgt-lang None
```

### 3. 或批量处理文件夹

```bash
# ./videos 下所有视频（递归扫描，已有 .srt 的自动跳过）
python batch_generate.py "./videos"

# 预览将要处理的文件
python batch_generate.py "./videos" --dry-run

# 自定义语言 / 设备
python batch_generate.py "./videos" --src-lang ja --tgt-lang English --asr-device GPU.1 --trans-device GPU
```

### 分步执行

上面的入口通过 subprocess 调用两个阶段，也可以直接运行：

```bash
python video_to_subtitle.py "input.mkv" --task transcribe --language zh   # 视频 → SRT
python translate_srt.py input.srt --tgt-lang 中文                          # SRT → 翻译后 SRT
```

完整参数见各脚本 `--help`。

## 性能实测

测试平台：Intel Core Ultra 7 265K + Arc B580。

**转录** —— Whisper Large-v3 FP16，5 分钟音频：

| 模式 | 设备 | batch_size | 耗时 | 分段数 |
|------|------|-----------|------|--------|
| VAD（默认） | Arc B580（`GPU.1`） | 1 | 30.0 s | 89 |
| VAD | Arc B580 | 2 | 29.5 s | 89 |
| 无 VAD | Arc B580 | 2 | 27.7 s | 22 |

2.5 小时电影约 4.4 分钟完成，产出 333 条字幕。

**翻译** —— Hy-MT2：

| 模型 | 设备 | 速度 | 占用 |
|------|------|------|------|
| 1.8B FP16 | Arc B580 | ~30+ tok/s | ~3.5 GB |
| 1.8B FP16 | CPU | ~10 tok/s | ~3.5 GB |
| 7B INT4 | Arc B580 | 7.85 tok/s | ~4.2 GB |
| 7B INT4 | CPU | 2.45 tok/s | ~4.2 GB |

## 设备说明

`GPU.1` 是独立 Arc 显卡，也是 ASR 的默认设备；`GPU.0` 是核显；`CPU` / `AUTO` 通用。

**NPU 对这两个模型都不可用。** Intel NPU 要求模型完全静态 shape，而 Whisper 和带 stateful KV cache 的 LLM 都含动态变长序列维度。请使用 GPU 或 CPU。

## 文档

| 文档 | 内容 |
|------|------|
| [`tools/model_conversion/README.zh-CN.md`](tools/model_conversion/README.zh-CN.md) | Hunyuan → OpenVINO 转换与推理指南 |
| [`models/SILERO_VAD/README.zh-CN.md`](models/SILERO_VAD/README.zh-CN.md) | Silero VAD 模型卡与缺失恢复 |
| [`models/whisper-large-v3-fp16-ov/README.md`](models/whisper-large-v3-fp16-ov/README.md) | Whisper 模型卡 |
| [`models/Hy-MT2-7B-ov-int4/README.zh-CN.md`](models/Hy-MT2-7B-ov-int4/README.zh-CN.md) | Hy-MT2 INT4 模型卡 |
