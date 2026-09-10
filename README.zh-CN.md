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

| 路径 | 作用 | 状态 |
|------|------|------|
| `tools/model_conversion/` | Hunyuan → OpenVINO 导出（FP16 / INT4）+ 推理验证 | 已上传 |
| `video_to_subtitle.py` | 视频/音频 → SRT（Whisper + VAD） | 待上传 |
| `translate_srt.py` | SRT → 翻译后 SRT（Hy-MT2） | 待上传 |
| `public/` | 独立分发包：一键与批量入口 | 待上传 |

本仓库分批上传，目前只有模型转换工具就位，其余部分在后续提交中补齐。

## 环境要求

Python 3.10；转录环节另需 [ffmpeg](https://ffmpeg.org/)。

```bash
conda create -n py310_openvino python=3.10
conda activate py310_openvino

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install openvino nncf optimum[openvino] transformers soundfile silero-vad
```

`nncf` 仅 INT4 量化需要，纯 FP16 导出可跳过。

## 快速开始

### 方式一：下载已转换好的模型

INT4 量化后的 7B 模型已发布到 ModelScope，开箱即用：

**[ahbencat/Hy-MT2-7B-ov-int4](https://modelscope.cn/models/ahbencat/Hy-MT2-7B-ov-int4)** —— Hy-MT2 7B，OpenVINO INT4，约 4.2 GB

```bash
pip install modelscope
modelscope download --model ahbencat/Hy-MT2-7B-ov-int4 --local_dir ./Hy-MT2-7B-ov-int4
```

之后用 `--model-dir ./Hy-MT2-7B-ov-int4` 指向它即可。

### 方式二：自行转换

```bash
cd tools/model_conversion

# Hy-MT2 1.8B -> OpenVINO FP16（stateful KV cache）
python convert_hunyuan_optimum.py

# Hy-MT2 7B -> OpenVINO INT4（~14 GB -> ~4.2 GB）
python convert_hunyuan_int4.py

# 验证导出结果
python infer_hunyuan_openvino.py --text "Hello, world." --tgt-lang "中文"
```

完整说明见 [`tools/model_conversion/README.zh-CN.md`](tools/model_conversion/README.zh-CN.md)。

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

## 关于路径

这些是为特定机器编写的独立脚本，不是打包好的库。模型目录和 ffmpeg 可执行文件都以 Windows 绝对路径硬编码在各脚本顶部。多数可通过命令行覆盖（`--model-dir`、`--output-dir`）；`FFMPEG_PATH` 不行，需要直接改脚本。

## 文档

| 文档 | 内容 |
|------|------|
| [`tools/model_conversion/README.zh-CN.md`](tools/model_conversion/README.zh-CN.md) | Hunyuan → OpenVINO 转换与推理指南 |
