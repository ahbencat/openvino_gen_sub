#!/usr/bin/env python3
"""
Hy-MT2-1.8B OpenVINO 推理测试
"""

from pathlib import Path
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer


MODEL_DIR = Path(r"D:\Projects\whisper_cpp_win\Hy-MT2-1.8B-ov")
DEVICE = "CPU"  # CPU / GPU / AUTO


def main():
    print("=" * 50)
    print("Loading OpenVINO model...")
    print(f"  Device: {DEVICE}")
    print(f"  Model:  {MODEL_DIR}")
    print()

    model = OVModelForCausalLM.from_pretrained(
        MODEL_DIR, device=DEVICE, use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    prompts = [
        "Hello, how are you?",
        "Explain what is artificial intelligence in one sentence.",
        "What is the capital of France?",
    ]

    for prompt in prompts:
        print("-" * 50)
        print(f"Prompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Result: {result}")
        print()

    print("=" * 50)
    print("Done!")


if __name__ == "__main__":
    main()