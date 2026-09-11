# OpenVINO Video Subtitle Generator

**English** | [简体中文](README.zh-CN.md)

End-to-end video subtitle generation and translation on Intel hardware, built on OpenVINO.

- **Transcribe** — Whisper Large-v3 FP16 turns video/audio into SRT, with Silero VAD cutting segments on natural pauses instead of a fixed clock
- **Translate** — Hy-MT2 / Hunyuan Dense V1 translates the SRT into any target language, timeline preserved verbatim
- **Convert** — export Hunyuan models to OpenVINO FP16 or INT4 yourself

Developed and measured on an Intel Core Ultra 7 265K + Arc B580.

## Pipeline

```
video.mkv
    |  ffmpeg  ->  16 kHz mono wav
    v
Silero VAD  ->  speech segments
    |
    |  Whisper Large-v3 FP16 (OpenVINO, Arc GPU)
    v
source.srt
    |
    |  Hy-MT2 / Hunyuan Dense V1 (OpenVINO)
    v
translated.srt
```

Whisper's own `translate` task only ever emits English. For any other target language you must transcribe first and then translate — that is the reason for the two stages.

## Components

| Path | Role |
|------|------|
| `generate_and_translate.py` | One-shot driver: video/audio → source SRT → translated SRT |
| `batch_generate.py` | Batch driver: scan a folder and process every video in it |
| `video_to_subtitle.py` | Video/audio → SRT (Whisper Large-v3 + Silero VAD) |
| `translate_srt.py` | SRT → translated SRT (Hy-MT2), timeline preserved |
| `tools/model_conversion/` | Hunyuan → OpenVINO export (FP16 / INT4) + inference checks |
| `models/` | Model cards; download the weights into these folders |

## Requirements

Python 3.10, plus [ffmpeg](https://ffmpeg.org/) available on `PATH` (or point the `FFMPEG_PATH` environment variable at the binary).

```bash
pip install -r requirements.txt
```

Or assemble the environment manually:

```bash
conda create -n py310_openvino python=3.10
conda activate py310_openvino

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install openvino nncf optimum[openvino] transformers soundfile silero-vad
```

`nncf` is only needed for INT4 quantization — skip it if you only download pre-converted models.

## Getting started

### 1. Get the models

Both models are needed; the weights are not committed to this repository — download or convert them into the `models/` folders so the script defaults find them:

**Whisper Large-v3 FP16 (OpenVINO)** — from [OpenVINO/whisper-large-v3-fp16-ov](https://huggingface.co/OpenVINO/whisper-large-v3-fp16-ov) (Apache-2.0):

```bash
hf download OpenVINO/whisper-large-v3-fp16-ov --local-dir models/whisper-large-v3-fp16-ov
```

**Hy-MT2** — download the pre-converted INT4 7B from ModelScope: **[ahbencat/Hy-MT2-7B-ov-int4](https://modelscope.cn/models/ahbencat/Hy-MT2-7B-ov-int4)** (~4.2 GB):

```bash
modelscope download --model ahbencat/Hy-MT2-7B-ov-int4 --local_dir models/Hy-MT2-7B-ov-int4
```

Or convert it yourself (1.8B FP16, or tweaked INT4 settings) — see [`tools/model_conversion/README.md`](tools/model_conversion/README.md).

Any other location works too; point the scripts at it with `--model-dir`.

### 2. Generate subtitles

```bash
# One file: transcribe + translate (defaults: target language 中文)
python generate_and_translate.py "input.mkv"

# Known source language, English output
python generate_and_translate.py "input.mkv" --src-lang ja --tgt-lang English

# Transcription only, no translation
python generate_and_translate.py "input.mkv" --tgt-lang None
```

### 3. Or batch-process a folder

```bash
# Every video under ./videos (recursive, skips ones with an existing .srt)
python batch_generate.py "./videos"

# Preview what would run
python batch_generate.py "./videos" --dry-run

# Custom languages / devices
python batch_generate.py "./videos" --src-lang ja --tgt-lang English --asr-device GPU.1 --trans-device GPU
```

### Stage by stage

The drivers above call the two stages via subprocess; you can also run them directly:

```bash
python video_to_subtitle.py "input.mkv" --task transcribe --language zh   # video → SRT
python translate_srt.py input.srt --tgt-lang 中文                          # SRT → translated SRT
```

See each script's `--help` for the full option list.

## Performance

Measured on Intel Core Ultra 7 265K + Arc B580.

**Transcription** — Whisper Large-v3 FP16, 5 minutes of audio:

| Mode | Device | batch_size | Time | Segments |
|------|--------|-----------|------|----------|
| VAD (default) | Arc B580 (`GPU.1`) | 1 | 30.0 s | 89 |
| VAD | Arc B580 | 2 | 29.5 s | 89 |
| No VAD | Arc B580 | 2 | 27.7 s | 22 |

A 2.5-hour movie completes in roughly 4.4 minutes and yields 333 subtitle entries.

**Translation** — Hy-MT2:

| Model | Device | Speed | Footprint |
|-------|--------|-------|-----------|
| 1.8B FP16 | Arc B580 | ~30+ tok/s | ~3.5 GB |
| 1.8B FP16 | CPU | ~10 tok/s | ~3.5 GB |
| 7B INT4 | Arc B580 | 7.85 tok/s | ~4.2 GB |
| 7B INT4 | CPU | 2.45 tok/s | ~4.2 GB |

## Devices

`GPU.1` is the discrete Arc GPU and the default for ASR, `GPU.0` is the integrated GPU, and `CPU` / `AUTO` work everywhere.

**NPU does not work for either model.** Intel's NPU requires fully static shapes, while both Whisper and the stateful-KV-cache LLM carry a dynamic sequence-length dimension. Use GPU or CPU.

## Documentation

| Document | Contents |
|----------|----------|
| [`tools/model_conversion/README.md`](tools/model_conversion/README.md) | Hunyuan → OpenVINO conversion and inference guide |
| [`models/SILERO_VAD/README.md`](models/SILERO_VAD/README.md) | Silero VAD model card and recovery steps |
| [`models/whisper-large-v3-fp16-ov/README.md`](models/whisper-large-v3-fp16-ov/README.md) | Whisper model card |
| [`models/Hy-MT2-7B-ov-int4/README.md`](models/Hy-MT2-7B-ov-int4/README.md) | Hy-MT2 INT4 model card |
