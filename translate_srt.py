#!/usr/bin/env python3
"""
SRT subtitle translation tool based on Hy-MT2 OpenVINO.

Reads an existing SRT file, translates subtitle text one by one or in
merged blocks, and keeps the original timeline untouched.

Usage:
  python translate_srt.py input.srt
  python translate_srt.py input.srt --tgt-lang 中文
  python translate_srt.py input.srt --model-dir models/Hy-MT2-7B-ov-int4 --device GPU
  python translate_srt.py input.srt --context 2
  python translate_srt.py input.srt -o output_translated.srt
"""

import argparse
import re
import time
from pathlib import Path

from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, logging as transformers_logging


SCRIPT_DIR = Path(__file__).parent
DEFAULT_TRANS_MODEL_DIR = SCRIPT_DIR / "models" / "Hy-MT2-7B-ov-int4"


# ========== SRT parsing / writing ==========

TIMESTAMP_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")


def _ts_to_secs(ts: str) -> float:
    m = TIMESTAMP_RE.match(ts)
    if not m:
        return 0.0
    h, mi, s, ms = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    return h * 3600 + mi * 60 + s + ms / 1000


def _secs_to_ts(secs: float) -> str:
    h = int(secs // 3600)
    mi = int((secs % 3600) // 60)
    s = int(secs % 60)
    ms = int(round((secs - int(secs)) * 1000))
    return f"{h:02d}:{mi:02d}:{s:02d},{ms:03d}"


def parse_srt(path: Path) -> list[dict]:
    """
    Parse an SRT file.
    Returns: [{"index": 1, "start": 0.0, "end": 5.54, "text": "..."}, ...]
    """
    text = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\n+", text.strip())
    segments = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        ts_match = re.match(
            r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})",
            lines[1],
        )
        if not ts_match:
            continue
        start = _ts_to_secs(ts_match[1])
        end = _ts_to_secs(ts_match[2])
        text = " ".join(l.strip() for l in lines[2:] if l.strip())
        segments.append({"index": idx, "start": start, "end": end, "text": text})
    return segments


def write_srt(segments: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"{seg['index']}\n")
            f.write(f"{_secs_to_ts(seg['start'])} --> {_secs_to_ts(seg['end'])}\n")
            f.write(f"{seg['translated']}\n\n")


# ========== Translation prompt ==========

def make_translate_prompt(source_text: str, target_lang: str = "English") -> str:
    if target_lang in ("English", "english", "en", "EN"):
        return (
            f"Translate the following text into {target_lang}. "
            f"Note that you should only output the translated result without any additional explanation:\n\n"
            f"{source_text}"
        )
    elif target_lang in ("中文", "chinese", "zh", "ZH", "Chinese"):
        return (
            f"将以下文本翻译为 {target_lang}，注意只需要输出翻译后的结果，不要额外解释：\n\n"
            f"{source_text}"
        )
    else:
        return (
            f"Translate the following text into {target_lang}. "
            f"Note that you should only output the translated result without any additional explanation:\n\n"
            f"{source_text}"
        )


# ========== Translation engine ==========

class Translator:
    """Hy-MT2 OpenVINO translation engine (reuses inference state, loads only once)."""

    def __init__(self, model_dir: str | Path, device: str):
        self.model_dir = Path(model_dir)
        self.device = device

        print(f"Loading model: {self.model_dir}")
        print(f"Device:       {device}")
        load_start = time.perf_counter()

        transformers_logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._load_chat_template()
        self.model = OVModelForCausalLM.from_pretrained(self.model_dir, device=device)

        load_time = time.perf_counter() - load_start
        print(f"Loaded in {load_time:.2f}s")
        print(f"  stateful: {getattr(self.model, 'stateful', None)}")
        print()

    def _load_chat_template(self):
        if self.tokenizer.chat_template is not None:
            return
        for p in (self.model_dir.parent / "chat_template.jinja",
                  self.model_dir / "chat_template.jinja"):
            if p.exists():
                self.tokenizer.chat_template = p.read_text(encoding="utf-8")
                break

    def translate(self, text: str, target_lang: str = "English", **gen_kwargs) -> str:
        if not text.strip():
            return ""

        prompt_str = make_translate_prompt(text, target_lang=target_lang)

        if self.tokenizer.chat_template:
            final_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_str}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            final_prompt = prompt_str

        inputs = self.tokenizer(final_prompt, return_tensors="pt")

        kw = {
            "max_new_tokens": gen_kwargs.get("max_new_tokens", 512),
            "do_sample": True,
            "temperature": gen_kwargs.get("temperature", 0.7),
            "top_p": gen_kwargs.get("top_p", 0.6),
            "top_k": gen_kwargs.get("top_k", 20),
            "repetition_penalty": gen_kwargs.get("repetition_penalty", 1.05),
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        outputs = self.model.generate(**inputs, **kw)
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Strip the input prompt from the result
        if result.startswith(final_prompt):
            result = result[len(final_prompt):].strip()
        else:
            input_decoded = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            if result.startswith(input_decoded):
                result = result[len(input_decoded):].strip()
            elif result.startswith(prompt_str):
                result = result[len(prompt_str):].strip()

        return result


# ========== Main flow ==========

def build_blocks(segments: list[dict], max_chars: int = 0) -> list[list[dict]]:
    """
    Merge adjacent short subtitles into translation blocks.
    Text inside each block is wrapped in <segN> tags so the model
    keeps them apart. max_chars=0 means one block per subtitle.
    """
    if max_chars <= 0:
        return [[s] for s in segments if s["text"].strip()]

    blocks = []
    current = []
    chars = 0
    for seg in segments:
        if not seg["text"].strip():
            continue
        if chars + len(seg["text"]) > max_chars and current:
            blocks.append(current)
            current = []
            chars = 0
        current.append(seg)
        chars += len(seg["text"])
    if current:
        blocks.append(current)
    return blocks


def translate_segments(
    translator: Translator,
    segments: list[dict],
    target_lang: str,
    merge_chars: int,
) -> list[dict]:
    for seg in segments:
        seg["translated"] = ""

    blocks = build_blocks(segments, max_chars=merge_chars)
    total = len(blocks)

    print(f"Translating {len(blocks)} block(s)...")
    print()

    for bi, block in enumerate(blocks, 1):
        # Build tagged text: <seg1>text</seg1> <seg2>text</seg2> ...
        tagged_lines = []
        for i, seg in enumerate(block):
            tagged_lines.append(f"<seg{i+1}>{seg['text']}</seg{i+1}>")
        combined = " ".join(tagged_lines)

        seg_chars = " + ".join(str(len(s["text"])) for s in block)
        print(f"  [{bi}/{total}] {len(block)} seg(s) [{seg_chars}] raw={sum(len(s['text']) for s in block)} tag={len(combined)} chars")
        t0 = time.perf_counter()
        translated = translator.translate(combined, target_lang=target_lang)
        elapsed = time.perf_counter() - t0

        # Parse tagged response: extract <segN>...</segN>
        tag_pattern = re.compile(r"<seg(\d+)>(.*?)</seg\1>", re.DOTALL)
        matches = tag_pattern.findall(translated)

        if matches:
            for idx_str, content in matches:
                idx = int(idx_str) - 1
                if idx < len(block):
                    block[idx]["translated"] = content.strip()
        else:
            # Fallback: split by lines or assign full text to first segment
            lines = [l.strip() for l in translated.strip().splitlines() if l.strip()]
            for i, seg in enumerate(block):
                seg["translated"] = lines[i] if i < len(lines) else translated.strip()

        # Show preview
        preview = " | ".join(s["translated"][:40] for s in block if s["translated"])
        print(f"    -> {preview[:100]}{'...' if len(preview) > 100 else ''}")
        print(f"    ({elapsed:.2f}s)")

    # Fallback: fill empty translations with original text
    for seg in segments:
        if not seg["translated"]:
            seg["translated"] = seg["text"]

    return segments


def main():
    parser = argparse.ArgumentParser(
        description="SRT 字幕翻译工具 - 基于 Hy-MT2 OpenVINO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认翻译为英文
  python translate_srt.py input.srt

  # 翻译为中文
  python translate_srt.py input.srt --tgt-lang 中文

  # 使用 INT4 7B 模型 + GPU
  python translate_srt.py input.srt --model-dir models/Hy-MT2-7B-ov-int4 --device GPU

  # 指定输出路径
  python translate_srt.py input.srt -o output.srt
        """,
    )

    parser.add_argument("input", type=Path, help="输入 SRT 文件路径")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="输出 SRT 路径（默认: 输入文件名 + _translated.srt）")

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_TRANS_MODEL_DIR,
        help="OpenVINO 模型路径 (默认: " + str(DEFAULT_TRANS_MODEL_DIR) + ")",
    )
    parser.add_argument(
        "--device",
        default="CPU",
        choices=["CPU", "GPU", "GPU.0", "GPU.1", "NPU", "AUTO"],
        help="推理设备 (默认: CPU)",
    )

    parser.add_argument(
        "--tgt-lang",
        default="English",
        help="目标语言 (默认: English)",
    )
    parser.add_argument(
        "--merge-chars",
        type=int,
        default=0,
        help="合并短字幕的最大字符数，0=逐条翻译 (默认: 0)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="最大生成长度 (默认: 512)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"SRT file not found: {args.input}")

    if args.output is None:
        args.output = args.input.with_stem(args.input.stem + "_translated")

    # 1. Parse the SRT
    print("=" * 60)
    print("SRT TRANSLATOR - Hy-MT2 OpenVINO")
    print("=" * 60)
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Target: {args.tgt_lang}")
    print()

    segments = parse_srt(args.input)
    if not segments:
        print("[Error] No valid subtitle segments found.")
        return

    text_segments = [s for s in segments if s["text"].strip()]
    print(f"Parsed {len(segments)} segments ({len(text_segments)} non-empty)")
    print()

    # 2. Load the translation model
    translator = Translator(args.model_dir, args.device)

    # 3. Translate
    translate_segments(
        translator,
        segments,
        target_lang=args.tgt_lang,
        merge_chars=args.merge_chars,
    )

    # 4. Write the SRT back
    write_srt(segments, args.output)
    print()
    print(f"[OK] Translated SRT saved: {args.output}")


if __name__ == "__main__":
    main()