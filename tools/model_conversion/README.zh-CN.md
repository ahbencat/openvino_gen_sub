# Hy-MT2 / Hunyuan Dense V1 OpenVINO 模型转换与推理指南

[English](README.md) | **简体中文**

## 目录

- [概述](#概述)
- [环境准备](#环境准备)
- [脚本一览](#脚本一览)
- [方法一：optimum-intel 导出 FP16（推荐）](#方法一optimum-intel-导出-fp16推荐)
- [方法二：NNCF INT4 量化（7B 模型压缩）](#方法二nncf-int4-量化7b-模型压缩)
- [推理测试](#推理测试)
- [性能对比](#性能对比)
- [常见问题](#常见问题)

---

## 概述

本项目支持将 **Hy-MT2**（腾讯混元翻译模型，含 1.8B 和 7B 两种规格）转换为 OpenVINO 格式，在 Intel 平台上高效运行。

支持的特性：

- **Stateful KV Cache**：optimum-intel 原生导出，64 个 ReadValue/Assign 内部状态节点
- **INT4 权重量化**：NNCF 混合精度压缩（83% INT4 + 17% INT8），14GB → 4.17GB
- **多设备推理**：CPU / Intel Arc GPU / Intel NPU
- **Chat Template**：自动加载 Hy-MT2 专有 `<|hy_begin▁of▁sentence|>` / `<|hy_User|>` / `<|hy_Assistant|>` 模板

---

## 环境准备

### Conda 环境

```bash
conda create -n py310_openvino python=3.10
conda activate py310_openvino
```

### 安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install openvino nncf
pip install optimum[openvino]
pip install transformers
```

> **注意**：`nncf` 仅 INT4 量化需要，纯 FP16 导出可跳过。

### 脚本目录结构

转换与推理脚本位于本仓库的 `tools/model_conversion/`：

```
<仓库根>/
└── tools/model_conversion/
    ├── convert_hunyuan_optimum.py   # FP16 导出
    ├── convert_hunyuan_int4.py      # INT4 量化
    ├── infer_hunyuan_openvino.py    # 推理脚本
    └── test_openvino_model.py       # 加载冒烟测试
```

### 模型目录结构

模型不随仓库分发，需自行下载并存放。脚本内的默认路径是开发机上的 Windows 绝对路径：

```
D:\Projects\whisper_cpp_win\
├── Hy-MT2-1.8B\              # 原始 PyTorch 1.8B 模型
├── Hy-MT2-7B\                # 原始 PyTorch 7B 模型
├── Hy-MT2-1.8B-ov-optimum\   # 导出后的 OpenVINO FP16 1.8B
└── Hy-MT2-7B-ov-int4\        # 导出后的 OpenVINO INT4 7B
```

> **换机器时**：用 `--model-dir` / `--output-dir` 命令行参数覆盖，或直接改脚本顶部的 `DEFAULT_MODEL_DIR` / `MODEL_DIR` 常量。`test_openvino_model.py` 没有命令行参数，只能改脚本。

---

## 脚本一览

| 脚本 | 用途 | 适用模型 |
|------|------|----------|
| `convert_hunyuan_optimum.py` | optimum-intel FP16 导出 + stateful KV cache | 1.8B / 7B |
| `convert_hunyuan_int4.py` | 两步：先 FP16 导出，再 NNCF INT4 权重量化 | 7B（推荐） |
| `infer_hunyuan_openvino.py` | 加载 OpenVINO 模型进行翻译推理 | 以上任意导出 |
| `test_openvino_model.py` | 最小加载 + 生成冒烟测试（无参数，改脚本内常量） | 以上任意导出 |

---

## 方法一：optimum-intel 导出 FP16（推荐）

使用 optimum-intel 的 `main_export()` 管道，一步完成 FP16 + stateful KV cache 导出。

### 命令

```bash
cd tools/model_conversion

# 导出 1.8B 模型（默认）
python convert_hunyuan_optimum.py

# 导出 7B 模型（指定路径）
python convert_hunyuan_optimum.py \
    --model-dir "D:\Projects\whisper_cpp_win\Hy-MT2-7B" \
    --output-dir "D:\Projects\whisper_cpp_win\Hy-MT2-7B-ov-fp16"
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-dir` | `Hy-MT2-1.8B` | 原始 PyTorch 模型路径 |
| `--output-dir` | `Hy-MT2-1.8B-ov-optimum` | OpenVINO 输出路径 |
| `--task` | `text-generation-with-past` | 导出任务（含 KV cache） |
| `--device` | `CPU` | 导出的参考设备（CPU/GPU） |
| `--fp32` | (关闭) | 导出 FP32 而非 FP16 |

### 输出文件

```
Hy-MT2-1.8B-ov-optimum/
├── openvino_model.xml      # 模型结构定义
├── openvino_model.bin      # 模型权重
├── config.json             # 模型配置
├── tokenizer.json          # Tokenizer
├── tokenizer_config.json   # Tokenizer 配置
└── chat_template.jinja     # 对话模板
```

### 原理说明

- 自动注册 `hunyuan_v1_dense` 模型类型到 optimum 的 `TasksManager`
- 复用 `Qwen2OpenVINOConfig`（二者架构一致）
- `stateful=True` 开启内部 KV cache（64 个状态节点 = 32 层 × key + value）
- `OVConfig(dtype="fp16")` 防止自动 INT8 量化

---

## 方法二：NNCF INT4 量化（7B 模型压缩）

两步完成：先用 optimum-intel 导出 FP16，再用 NNCF 做 INT4 权重量化。

适合将 7B 模型从 ~14GB 压缩到 ~4GB。

### 命令

```bash
cd tools/model_conversion

python convert_hunyuan_int4.py
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `CPU` | FP16 导出的参考设备 |
| `--no-clean` | (关闭) | 不删除已有的 FP16 临时目录 |

### INT4 量化细节

```python
nncf.compress_weights(
    model,
    mode=nncf.CompressWeightsMode.INT4_SYM,  # 对称 INT4
    group_size=128,                            # 分组大小
    ratio=0.9,                                 # 90% INT4 + 10% INT8
)
```

- 混合精度：149/224 层 INT4 + 75/224 层 INT8（重要层保留 INT8）
- 压缩效果：14 GB → 4.17 GB（~70% 压缩率）

### 注意事项

- **Rich/GBK 编码问题**：Windows 中文控制台（代码页 936）无法显示 Rich 进度条的 `•` 字符。脚本已自动处理：

  ```python
  import rich._windows_renderer
  rich._windows_renderer.legacy_windows_render = lambda buffer, term: None
  ```

- **Windows .bin 文件锁定**：`core.read_model()` 会内存映射 .bin 文件，同一目录下无法覆盖保存。脚本使用临时目录保存 INT4 结果后 `shutil.move()` 到目标位置。
- **跨盘符移动**：使用 `shutil.move()` 而非 `Path.rename()` 以支持 C: → D: 的跨盘移动。

---

## 推理测试

### 基本用法

```bash
cd tools/model_conversion

# 默认使用 Hy-MT2-1.8B-ov-optimum（CPU）
python infer_hunyuan_openvino.py

# 指定 INT4 7B 模型 + GPU
python infer_hunyuan_openvino.py \
    --model-dir "D:\Projects\whisper_cpp_win\Hy-MT2-7B-ov-int4" \
    --device GPU

# 翻译句子
python infer_hunyuan_openvino.py \
    --text "The global economy is facing unprecedented challenges." \
    --tgt-lang "中文"

# 翻译到英文
python infer_hunyuan_openvino.py \
    --text "全球经济正面临前所未有的挑战。" \
    --tgt-lang "English"

# 使用自定义 prompt（跳过自动翻译 prompt 构建）
python infer_hunyuan_openvino.py \
    --prompt "Summarize the following text: ..." \
    --max-new-tokens 256

# 不使用 chat template（直接输入原始 prompt）
python infer_hunyuan_openvino.py \
    --text "Hello, how are you?" \
    --tgt-lang "中文" \
    --no-chat-template
```

### 完整参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-dir` | `Hy-MT2-1.8B-ov-optimum` | OpenVINO 模型路径 |
| `--device` | `CPU` | 推理设备（CPU / GPU / NPU / AUTO） |
| `--text` | (示例文本) | 待翻译文本 |
| `--src-lang` | 自动 | 源语言 |
| `--tgt-lang` | `English` | 目标语言 |
| `--prompt` | (无) | 自定义 prompt，跳过翻译 prompt 构建 |
| `--no-chat-template` | (关闭) | 不使用 chat template，直接输入 |
| `--max-new-tokens` | `512` | 最大生成长度 |
| `--temperature` | `0.7` | 采样温度 |
| `--top-p` | `0.6` | Top-p 采样 |
| `--top-k` | `20` | Top-k 采样 |
| `--repetition-penalty` | `1.05` | 重复惩罚系数 |

### 翻译 Prompt 格式

脚本自动构建 Hy-MT2 官方推荐的翻译 prompt：

**中文 → 英文：**
```
Translate the following text into English. Note that you should only output the translated result without any additional explanation:

{source_text}
```

**英文 → 中文：**
```
将以下文本翻译为 中文，注意只需要输出翻译后的结果，不要额外解释：

{source_text}
```

### 输出示例

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

### 快速冒烟测试

只想确认模型能加载并生成，用 `test_openvino_model.py`。它没有命令行参数，改脚本顶部的 `MODEL_DIR` / `DEVICE` 常量即可：

```bash
python test_openvino_model.py
```

---

## 性能对比

测试基于 Intel Core Ultra 7 265K + Arc B580 平台：

| 模型 | 设备 | 速度 | 显存占用 | 备注 |
|------|------|------|----------|------|
| 1.8B FP16 | CPU | ~10 tok/s | ~3.5 GB | 轻量快速 |
| 1.8B FP16 | GPU (Arc B580) | ~30+ tok/s | ~3.5 GB | 最快选项 |
| 7B INT4 | CPU | 2.45 tok/s | ~4.2 GB | 离线可用 |
| 7B INT4 | GPU (Arc B580) | 7.85 tok/s | ~4.2 GB | 推荐配置 |
| 7B INT4 | NPU (AI Boost) | — | — | 不支持动态 shape |

**NPU 限制说明**：Intel NPU 要求模型完全静态 shape，而 stateful KV cache 模型包含动态变长序列维度。NPU 适用于固定尺寸的小模型，不支持大语言模型的变长推理。

---

## 常见问题

### Q: 转换时报错 `'hunyuan_v1_dense' is not registered`

optimum-intel 官方未内置 `hunyuan_v1_dense` 的 OpenVINO 导出配置。脚本通过 `TasksManager.create_register()` 在运行时注册，复用 `Qwen2OpenVINOConfig`。确保使用 `convert_hunyuan_optimum.py` 而非直接调用 optimum CLI。

### Q: Rich 进度条报 `UnicodeEncodeError: 'gbk' codec can't encode character`

Windows 中文控制台（代码页 936）无法显示 Rich 的特殊字符。解决方法已在 `convert_hunyuan_int4.py` 中内置：

```bash
# 如果手动运行 NNCF，添加环境变量
set PYTHONIOENCODING=utf-8
```

### Q: OpenVINO 编译模型时报 `to_shape was called on a dynamic shape`

模型包含动态维度。在 NPU 设备上编译需要静态 shape，换用 CPU 或 GPU 即可。

### Q: GPU 推理时加载很慢（~18秒）

这是 OpenVINO 在 GPU 上的模型编译时间，与模型大小成正比。模型编译后会缓存，后续加载会更快。推理速度不受影响。

### Q: 如何查看 OpenVINO 可用设备？

```python
import openvino as ov
core = ov.Core()
for d in core.available_devices:
    print(d, core.get_property(d, "FULL_DEVICE_NAME"))
```