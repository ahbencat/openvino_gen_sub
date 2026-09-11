# Hy-MT2-7B-ov-int4

**English** | [简体中文](README.zh-CN.md)

[Hy-MT2](https://huggingface.co/tencent/Hy-MT2-7B) (Tencent Hunyuan translation model, 7B) converted to the OpenVINO™ IR format with NNCF INT4 weight compression.

- **Homepage**: <https://modelscope.cn/models/ahbencat/Hy-MT2-7B-ov-int4>
- **Model creator**: [Tencent](https://huggingface.co/tencent) — original model [tencent/Hy-MT2-7B](https://huggingface.co/tencent/Hy-MT2-7B)
- **Converter**: exported with [optimum-intel](https://huggingface.co/docs/optimum/intel/index) + NNCF by the scripts in [`tools/model_conversion/`](../../tools/model_conversion/README.md)

## Quantization

| | |
|---|---|
| Scheme | symmetric INT4, group size 128 |
| Mix | 149/224 layers INT4 + 75/224 layers INT8 (sensitive layers stay INT8) |
| Size | ~4.2 GB (from ~14 GB FP16) |
| KV cache | stateful (internal ReadValue/Assign states, no KV tensors on the graph) |

## Files

| File | Purpose |
|------|---------|
| `openvino_model.xml` / `openvino_model.bin` | OpenVINO IR model (INT4-compressed weights) |
| `config.json` | architecture config (`hunyuan_v1_dense`) |
| `generation_config.json` | recommended sampling parameters |
| `tokenizer.json` / `tokenizer_config.json` / `special_tokens_map.json` | tokenizer |
| `chat_template.jinja` | Hy-MT2 chat template |

## Compatibility

- OpenVINO ≥ 2025.2 (INT4 weights decompress on load; GPU/CPU both work)
- optimum-intel ≥ 1.23 (`OVModelForCausalLM`)
- **NPU is not supported** — the stateful model has a dynamic sequence dimension, while the NPU requires fully static shapes

## Usage

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

Recommended sampling parameters (already in `generation_config.json`): temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05.

Command-line translation check: see [`tools/model_conversion/infer_hunyuan_openvino.py`](../../tools/model_conversion/README.md#inference-testing).

## Performance

Measured on Intel Core Ultra 7 265K + Arc B580:

| Device | Speed | Footprint |
|--------|-------|-----------|
| Arc B580 (GPU) | 7.85 tok/s | ~4.2 GB |
| CPU | 2.45 tok/s | ~4.2 GB |
| NPU | — not supported — | |

## Legal information

The original model is distributed under its own license — see the [original model card](https://huggingface.co/tencent/Hy-MT2-7B). This converted copy inherits those terms.
