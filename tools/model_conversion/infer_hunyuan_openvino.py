#!/usr/bin/env python3
"""
Hy-MT2 / Hunyuan Dense V1 OpenVINO 翻译推理脚本

默认使用 optimum-intel 导出的 stateful KV cache 模型：
D:\Projects\whisper_cpp_win\Hy-MT2-1.8B-ov-optimum
"""

import argparse
import json
import time
from pathlib import Path

from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, logging


DEFAULT_MODEL_DIR = Path(r"D:\Projects\whisper_cpp_win\Hy-MT2-1.8B-ov-optimum")

# Hy-MT2 官方推荐推理参数
RECOMMENDED = {
    "temperature": 0.7,
    "top_p": 0.6,
    "top_k": 20,
    "repetition_penalty": 1.05,
}


def make_translate_prompt(
    source_text: str,
    target_lang: str = "English",
    source_lang: str = "中文",
    style: str | None = None,
) -> str:
    """基于 README 构建 Hy-MT2 翻译 prompt。"""
    if target_lang in ("English", "english", "en", "EN"):
        prompt = (
            f"Translate the following text into {target_lang}. "
            f"Note that you should only output the translated result without any additional explanation:\n\n"
            f"{source_text}"
        )
    elif target_lang in ("中文", "chinese", "zh", "ZH", "Chinese"):
        prompt = (
            f"将以下文本翻译为 {target_lang}，注意只需要输出翻译后的结果，不要额外解释：\n\n"
            f"{source_text}"
        )
    else:
        prompt = (
            f"Translate the following text into {target_lang}. "
            f"Note that you should only output the translated result without any additional explanation:\n\n"
            f"{source_text}"
        )
    return prompt


def load_chat_template(tokenizer, model_dir: Path):
    """从 chat_template.jinja 加载 chat template（如果 tokenizer 没有内置）。"""
    if tokenizer.chat_template is not None:
        return
    jinja = model_dir.parent / "chat_template.jinja"
    if not jinja.exists():
        jinja = model_dir / "chat_template.jinja"
    if jinja.exists():
        with open(jinja, encoding="utf-8") as f:
            tokenizer.chat_template = f.read()


def main():
    parser = argparse.ArgumentParser(description="Hy-MT2 OpenVINO 翻译推理")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU", "NPU", "AUTO"])
    parser.add_argument("--text", default=None, help="待翻译文本")
    parser.add_argument("--src-lang", default=None, help="源语言")
    parser.add_argument("--tgt-lang", default="English", help="目标语言")
    parser.add_argument("--prompt", default=None, help="自定义 prompt（跳过翻译 prompt 构建）")
    parser.add_argument("--no-chat-template", action="store_true", help="不使用 chat template，直接输入 prompt")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=RECOMMENDED["temperature"])
    parser.add_argument("--top-p", type=float, default=RECOMMENDED["top_p"])
    parser.add_argument("--top-k", type=int, default=RECOMMENDED["top_k"])
    parser.add_argument("--repetition-penalty", type=float, default=RECOMMENDED["repetition_penalty"])
    args = parser.parse_args()

    logging.set_verbosity_error()

    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    print("=" * 60)
    print("Hy-MT2 OpenVINO 翻译推理")
    print("=" * 60)
    print(f"  Model:  {args.model_dir}")
    print(f"  Device: {args.device}")
    print()

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    load_chat_template(tokenizer, args.model_dir)
    model = OVModelForCausalLM.from_pretrained(args.model_dir, device=args.device)
    load_time = time.perf_counter() - load_start
    print(f"Loaded in {load_time:.2f}s")
    print(f"  use_cache: {model.use_cache}")
    print(f"  stateful:  {getattr(model, 'stateful', None)}")
    print()

    # 构建 prompt
    if args.prompt:
        prompt_str = args.prompt
    elif args.text:
        prompt_str = make_translate_prompt(args.text, target_lang=args.tgt_lang)
    else:
        prompt_str = make_translate_prompt(
            "The quick brown fox jumps over the lazy dog. 人工智能正在改变世界。",
            target_lang=args.tgt_lang,
        )

    if args.no_chat_template:
        final_prompt = prompt_str
    else:
        messages = [{"role": "user", "content": prompt_str}]
        if tokenizer.chat_template:
            final_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            final_prompt = prompt_str

    inputs = tokenizer(final_prompt, return_tensors="pt")

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }

    print("Input:" if args.text else "Prompt:")
    print(args.text or args.prompt or prompt_str)
    print()
    print(f"Translating to {args.tgt_lang}...")

    gen_start = time.perf_counter()
    outputs = model.generate(**inputs, **gen_kwargs)
    gen_time = time.perf_counter() - gen_start

    input_tokens = inputs["input_ids"].shape[-1]
    output_tokens = outputs.shape[-1]
    new_tokens = output_tokens - input_tokens
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print()
    print("Translation:")
    print(result)
    print()
    print("Stats:")
    print(f"  Input tokens: {input_tokens}")
    print(f"  New tokens:   {new_tokens}")
    print(f"  Time:         {gen_time:.2f}s")
    if gen_time > 0:
        print(f"  Speed:        {new_tokens / gen_time:.2f} tok/s")


if __name__ == "__main__":
    main()