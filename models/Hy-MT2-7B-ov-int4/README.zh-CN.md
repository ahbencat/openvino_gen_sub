# Hy-MT2-7B-ov-int4

[English](README.md) | **简体中文**

[Hy-MT2](https://huggingface.co/tencent/Hy-MT2-7B)（腾讯混元翻译模型，7B）转换为 OpenVINO™ IR 格式，并经 NNCF INT4 权重压缩。

- **模型主页**：<https://modelscope.cn/models/ahbencat/Hy-MT2-7B-ov-int4>
- **模型作者**：[Tencent](https://huggingface.co/tencent) —— 原始模型 [tencent/Hy-MT2-7B](https://huggingface.co/tencent/Hy-MT2-7B)
- **转换工具**：由仓库内 [`tools/model_conversion/`](../../tools/model_conversion/README.zh-CN.md) 的脚本使用 optimum-intel + NNCF 导出

## 量化信息

| | |
|---|---|
| 方案 | 对称 INT4，group size 128 |
| 混合精度 | 149/224 层 INT4 + 75/224 层 INT8（敏感层保留 INT8） |
| 体积 | 约 4.2 GB（FP16 约 14 GB） |
| KV cache | stateful（内部 ReadValue/Assign 状态，图上无 KV 张量） |

## 文件清单

| 文件 | 用途 |
|------|------|
| `openvino_model.xml` / `openvino_model.bin` | OpenVINO IR 模型（INT4 压缩权重） |
| `config.json` | 架构配置（`hunyuan_v1_dense`） |
| `generation_config.json` | 推荐采样参数 |
| `tokenizer.json` / `tokenizer_config.json` / `special_tokens_map.json` | 分词器 |
| `chat_template.jinja` | Hy-MT2 对话模板 |

## 兼容性

- OpenVINO ≥ 2025.2（INT4 权重加载时解压；GPU/CPU 均可）
- optimum-intel ≥ 1.23（`OVModelForCausalLM`）
- **不支持 NPU** —— stateful 模型含动态序列维度，而 NPU 要求完全静态 shape

## 用法

```python
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

model_dir = "Hy-MT2-7B-ov-int4"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = OVModelForCausalLM.from_pretrained(model_dir, device="GPU")

prompt = "Translate the following text into Chinese. Note that you should only output the translated result without any additional explanation:\n\nThe global economy is facing unprecedented challenges."
messages = [{"role": "user", "content": prompt}]

inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
)
outputs = model.generate(inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))
```

推荐采样参数（已写入 `generation_config.json`）：temperature 0.7、top_p 0.8、top_k 20、repetition_penalty 1.05。

命令行翻译验证：见 [`tools/model_conversion/infer_hunyuan_openvino.py`](../../tools/model_conversion/README.zh-CN.md#推理测试)。

## 性能

实测平台 Intel Core Ultra 7 265K + Arc B580：

| 设备 | 速度 | 占用 |
|------|------|------|
| Arc B580 (GPU) | 7.85 tok/s | ~4.2 GB |
| CPU | 2.45 tok/s | ~4.2 GB |
| NPU | —— 不支持 —— | |

## 法律信息

原始模型按其自身的许可协议分发 —— 详见[原始模型卡](https://huggingface.co/tencent/Hy-MT2-7B)。本转换副本沿用该协议。
