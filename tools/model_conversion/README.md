# Hy-MT2 / Hunyuan Dense V1 → OpenVINO: Conversion & Inference Guide

**English** | [简体中文](README.zh-CN.md)

## Contents

- [Overview](#overview)
- [Setup](#setup)
- [Scripts](#scripts)
- [Method 1: FP16 export via optimum-intel (recommended)](#method-1-fp16-export-via-optimum-intel-recommended)
- [Method 2: NNCF INT4 quantization (compressing the 7B)](#method-2-nncf-int4-quantization-compressing-the-7b)
- [Inference testing](#inference-testing)
- [Performance](#performance)
- [FAQ](#faq)

---

## Overview

These scripts convert **Hy-MT2** (Tencent's Hunyuan translation model, available in 1.8B and 7B) to OpenVINO so it runs efficiently on Intel hardware.

Supported:

- **Stateful KV cache** — native optimum-intel export, 64 internal ReadValue/Assign state nodes
- **INT4 weight quantization** — NNCF mixed precision (83% INT4 + 17% INT8), 14 GB → 4.17 GB
- **Multi-device inference** — CPU / Intel Arc GPU / Intel NPU
- **Chat template** — automatically loads Hy-MT2's own `<|hy_begin▁of▁sentence|>` / `<|hy_User|>` / `<|hy_Assistant|>` template

---

## Setup

### Conda environment

```bash
conda create -n py310_openvino python=3.10
conda activate py310_openvino
```

### Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install openvino nncf
pip install optimum[openvino]
pip install transformers
```

> **Note**: `nncf` is only needed for INT4 quantization — skip it for plain FP16 export.

### Script layout

The conversion and inference scripts live in `tools/model_conversion/`:

```
<repo root>/
└── tools/model_conversion/
    ├── convert_hunyuan_optimum.py   # FP16 export
    ├── convert_hunyuan_int4.py      # INT4 quantization
    ├── infer_hunyuan_openvino.py    # inference
    └── test_openvino_model.py       # load smoke test
```

### Model layout

Models are not distributed with this repository — download and place them yourself. The defaults baked into the scripts are absolute Windows paths from the development machine:

```
D:\Projects\whisper_cpp_win\
├── Hy-MT2-1.8B\              # original PyTorch 1.8B
├── Hy-MT2-7B\                # original PyTorch 7B
├── Hy-MT2-1.8B-ov-optimum\   # exported OpenVINO FP16 1.8B
└── Hy-MT2-7B-ov-int4\        # exported OpenVINO INT4 7B
```

> **On a different machine**: override with the `--model-dir` / `--output-dir` command-line options, or edit the `DEFAULT_MODEL_DIR` / `MODEL_DIR` constants at the top of each script. `test_openvino_model.py` takes no arguments — you have to edit it.

---

## Scripts

| Script | Purpose | Models |
|--------|---------|--------|
| `convert_hunyuan_optimum.py` | optimum-intel FP16 export + stateful KV cache | 1.8B / 7B |
| `convert_hunyuan_int4.py` | Two steps: FP16 export, then NNCF INT4 weight quantization | 7B (recommended) |
| `infer_hunyuan_openvino.py` | Load an OpenVINO model and run translation inference | any of the above |
| `test_openvino_model.py` | Minimal load + generate smoke test (no arguments, edit the constants) | any of the above |

---

## Method 1: FP16 export via optimum-intel (recommended)

Uses optimum-intel's `main_export()` pipeline to produce FP16 with a stateful KV cache in one step.

### Commands

```bash
cd tools/model_conversion

# Export the 1.8B (default)
python convert_hunyuan_optimum.py

# Export the 7B (explicit paths)
python convert_hunyuan_optimum.py \
    --model-dir "D:\Projects\whisper_cpp_win\Hy-MT2-7B" \
    --output-dir "D:\Projects\whisper_cpp_win\Hy-MT2-7B-ov-fp16"
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-dir` | `Hy-MT2-1.8B` | Original PyTorch model path |
| `--output-dir` | `Hy-MT2-1.8B-ov-optimum` | OpenVINO output path |
| `--task` | `text-generation-with-past` | Export task (includes KV cache) |
| `--device` | `CPU` | Reference device for the export (CPU/GPU) |
| `--fp32` | (off) | Export FP32 instead of FP16 |

### Output files

```
Hy-MT2-1.8B-ov-optimum/
├── openvino_model.xml      # model topology
├── openvino_model.bin      # weights
├── config.json             # model config
├── tokenizer.json          # tokenizer
├── tokenizer_config.json   # tokenizer config
└── chat_template.jinja     # chat template
```

### How it works

- Registers the `hunyuan_v1_dense` model type with optimum's `TasksManager` at runtime
- Reuses `Qwen2OpenVINOConfig` (the two architectures match)
- `stateful=True` enables the internal KV cache (64 state nodes = 32 layers × key + value)
- `OVConfig(dtype="fp16")` prevents automatic INT8 quantization

---

## Method 2: NNCF INT4 quantization (compressing the 7B)

Two steps: export FP16 with optimum-intel, then apply NNCF INT4 weight quantization.

Intended for taking the 7B from ~14 GB down to ~4 GB.

### Commands

```bash
cd tools/model_conversion

python convert_hunyuan_int4.py
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--device` | `CPU` | Reference device for the FP16 export |
| `--no-clean` | (off) | Keep an existing FP16 temp directory instead of deleting it |

### Quantization details

```python
nncf.compress_weights(
    model,
    mode=nncf.CompressWeightsMode.INT4_SYM,  # symmetric INT4
    group_size=128,                            # group size
    ratio=0.9,                                 # 90% INT4 + 10% INT8
)
```

- Mixed precision: 149/224 layers INT4 + 75/224 layers INT8 (sensitive layers stay INT8)
- Result: 14 GB → 4.17 GB (~70% reduction)

### Gotchas

- **Rich/GBK encoding**: a Chinese Windows console (code page 936) cannot render the `•` character in Rich progress bars. The script already handles this:

  ```python
  import rich._windows_renderer
  rich._windows_renderer.legacy_windows_render = lambda buffer, term: None
  ```

- **Windows .bin file locking**: `core.read_model()` memory-maps the .bin file, so you cannot save over it in the same directory. The script writes the INT4 result to a temp directory and then `shutil.move()`s it into place.
- **Cross-drive moves**: `shutil.move()` rather than `Path.rename()`, so C: → D: works.

---

## Inference testing

### Basic usage

```bash
cd tools/model_conversion

# Defaults to Hy-MT2-1.8B-ov-optimum on CPU
python infer_hunyuan_openvino.py

# INT4 7B on GPU
python infer_hunyuan_openvino.py \
    --model-dir "D:\Projects\whisper_cpp_win\Hy-MT2-7B-ov-int4" \
    --device GPU

# Translate a sentence
python infer_hunyuan_openvino.py \
    --text "The global economy is facing unprecedented challenges." \
    --tgt-lang "中文"

# Translate into English
python infer_hunyuan_openvino.py \
    --text "全球经济正面临前所未有的挑战。" \
    --tgt-lang "English"

# Custom prompt (skips the built-in translation prompt)
python infer_hunyuan_openvino.py \
    --prompt "Summarize the following text: ..." \
    --max-new-tokens 256

# Skip the chat template (feed the raw prompt)
python infer_hunyuan_openvino.py \
    --text "Hello, how are you?" \
    --tgt-lang "中文" \
    --no-chat-template
```

### Full options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-dir` | `Hy-MT2-1.8B-ov-optimum` | OpenVINO model path |
| `--device` | `CPU` | Inference device (CPU / GPU / NPU / AUTO) |
| `--text` | (sample text) | Text to translate |
| `--src-lang` | auto | Source language |
| `--tgt-lang` | `English` | Target language |
| `--prompt` | (none) | Custom prompt, bypasses translation prompt construction |
| `--no-chat-template` | (off) | Feed input directly without the chat template |
| `--max-new-tokens` | `512` | Maximum generation length |
| `--temperature` | `0.7` | Sampling temperature |
| `--top-p` | `0.6` | Top-p sampling |
| `--top-k` | `20` | Top-k sampling |
| `--repetition-penalty` | `1.05` | Repetition penalty |

### Translation prompt format

The script builds Hy-MT2's officially recommended translation prompt:

**Chinese → English:**
```
Translate the following text into English. Note that you should only output the translated result without any additional explanation:

{source_text}
```

**English → Chinese:**
```
将以下文本翻译为 中文，注意只需要输出翻译后的结果，不要额外解释：

{source_text}
```

### Sample output

```
============================================================
Hy-MT2 OpenVINO 翻译推理
============================================================
  Model:  D:\Projects\whisper_cpp_win\Hy-MT2-7B-ov-int4
  Device: GPU

Loaded in 18.03s
  use_cache: True
  stateful:  True

Input:
The global economy is facing unprecedented challenges.

Translating to 中文...

Translation:
全球经济正面临前所未有的挑战。

Stats:
  Input tokens: 37
  New tokens:   14
  Time:         1.27s
  Speed:        7.85 tok/s
```

### Quick smoke test

To just confirm a model loads and generates, use `test_openvino_model.py`. It takes no command-line arguments — edit the `MODEL_DIR` / `DEVICE` constants at the top:

```bash
python test_openvino_model.py
```

---

## Performance

Measured on an Intel Core Ultra 7 265K + Arc B580:

| Model | Device | Speed | Footprint | Notes |
|-------|--------|-------|-----------|-------|
| 1.8B FP16 | CPU | ~10 tok/s | ~3.5 GB | light and quick |
| 1.8B FP16 | GPU (Arc B580) | ~30+ tok/s | ~3.5 GB | fastest option |
| 7B INT4 | CPU | 2.45 tok/s | ~4.2 GB | works offline |
| 7B INT4 | GPU (Arc B580) | 7.85 tok/s | ~4.2 GB | recommended |
| 7B INT4 | NPU (AI Boost) | — | — | dynamic shapes unsupported |

**On the NPU limitation**: Intel's NPU requires fully static shapes, while a stateful-KV-cache model carries a dynamic variable-length sequence dimension. The NPU suits small fixed-size models; it cannot do variable-length LLM inference.

---

## FAQ

### Q: Conversion fails with `'hunyuan_v1_dense' is not registered`

optimum-intel does not ship an OpenVINO export config for `hunyuan_v1_dense`. The scripts register one at runtime via `TasksManager.create_register()`, reusing `Qwen2OpenVINOConfig`. Make sure you run `convert_hunyuan_optimum.py` rather than calling the optimum CLI directly.

### Q: Rich progress bar throws `UnicodeEncodeError: 'gbk' codec can't encode character`

A Chinese Windows console (code page 936) cannot render Rich's special characters. `convert_hunyuan_int4.py` already works around this:

```bash
# If you run NNCF manually, set the environment variable
set PYTHONIOENCODING=utf-8
```

### Q: OpenVINO reports `to_shape was called on a dynamic shape` when compiling

The model contains dynamic dimensions. Compiling for the NPU requires static shapes — switch to CPU or GPU.

### Q: GPU inference takes ~18 s to load

That is OpenVINO's model compilation time on the GPU, proportional to model size. Compiled models are cached, so subsequent loads are faster. Inference speed is unaffected.

### Q: How do I list the available OpenVINO devices?

```python
import openvino as ov
core = ov.Core()
for d in core.available_devices:
    print(d, core.get_property(d, "FULL_DEVICE_NAME"))
```
