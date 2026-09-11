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

| Path | Role | Status |
|------|------|--------|
| `tools/model_conversion/` | Hunyuan → OpenVINO export (FP16 / INT4) + inference checks | Available |
| `video_to_subtitle.py` | Video/audio → SRT (Whisper + VAD) | Not yet uploaded |
| `translate_srt.py` | SRT → translated SRT (Hy-MT2) | Not yet uploaded |
| `public/` | Self-contained package: one-shot and batch drivers | Not yet uploaded |

This repository is being published in stages. Only the model conversion tooling is in place so far; the rest lands in follow-up commits.

## Requirements

Python 3.10, plus [ffmpeg](https://ffmpeg.org/) on the transcription side.

```bash
conda create -n py310_openvino python=3.10
conda activate py310_openvino

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install openvino nncf optimum[openvino] transformers soundfile silero-vad
```

`nncf` is only needed for INT4 quantization — skip it for plain FP16 export.

## Getting started

Both models are needed; neither is committed to this repository — download or convert them yourself.

### Models

**Whisper Large-v3 FP16 (OpenVINO)** — for transcription, get it from [OpenVINO/whisper-large-v3-fp16-ov](https://huggingface.co/OpenVINO/whisper-large-v3-fp16-ov) (Apache-2.0):

```bash
pip install huggingface_hub
hf download OpenVINO/whisper-large-v3-fp16-ov --local-dir ./whisper-large-v3-fp16-ov
```

The card in [`models/whisper-large-v3-fp16-ov/`](models/whisper-large-v3-fp16-ov/README.md) documents compatibility (OpenVINO ≥ 2025.2, Optimum Intel ≥ 1.23) and inference examples.

**Hy-MT2 (OpenVINO)** — for translation, two routes:

- **Download the pre-converted INT4 7B** — published on ModelScope, works out of the box: **[ahbencat/Hy-MT2-7B-ov-int4](https://modelscope.cn/models/ahbencat/Hy-MT2-7B-ov-int4)** (~4.2 GB)

  ```bash
  pip install modelscope
  modelscope download --model ahbencat/Hy-MT2-7B-ov-int4 --local_dir ./Hy-MT2-7B-ov-int4
  ```

- **Convert it yourself** (1.8B FP16, or tweaked INT4 settings):

  ```bash
  cd tools/model_conversion

  # Hy-MT2 1.8B -> OpenVINO FP16 (stateful KV cache)
  python convert_hunyuan_optimum.py

  # Hy-MT2 7B -> OpenVINO INT4 (~14 GB -> ~4.2 GB)
  python convert_hunyuan_int4.py

  # Verify the export
  python infer_hunyuan_openvino.py --text "Hello, world." --tgt-lang "中文"
  ```

  See [`tools/model_conversion/README.md`](tools/model_conversion/README.md) for the full guide.

Point the scripts at wherever you put them with `--model-dir`.

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

## A note on paths

These are standalone scripts written for one specific machine, not a packaged library. Model directories and the ffmpeg binary are hardcoded as absolute Windows paths at the top of each script. Most are overridable on the command line (`--model-dir`, `--output-dir`); `FFMPEG_PATH` is not — edit the script.

## Documentation

| Document | Contents |
|----------|----------|
| [`tools/model_conversion/README.md`](tools/model_conversion/README.md) | Hunyuan → OpenVINO conversion and inference guide |
