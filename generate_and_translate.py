#!/usr/bin/env python3
"""
One-shot subtitle generation + translation.

Usage:
    python generate_and_translate.py "input.mkv"
    python generate_and_translate.py "input.mkv" --tgt-lang English
    python generate_and_translate.py "input.mkv" --src-lang ja --tgt-lang 中文 --asr-device GPU.1 --trans-device GPU

How it works:
    1. video_to_subtitle.py generates source-language subtitles (transcribe mode)
    2. translate_srt.py translates them to the target language
    3. The final SRT is written next to the input file
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
DEFAULT_ASR_MODEL_DIR = SCRIPT_DIR / "models" / "whisper-large-v3-fp16-ov"
DEFAULT_TRANS_MODEL_DIR = SCRIPT_DIR / "models" / "Hy-MT2-7B-ov-int4"

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".mpeg", ".mpg"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aac", ".m4a", ".ogg", ".wma"}


def run_step(cmd, desc):
    """Run a subcommand and check its return code."""
    print("\n" + "=" * 60)
    print(f"[{desc}]")
    print("=" * 60)
    print(" ".join(str(c) for c in cmd))
    print()

    start = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(f"{desc} 失败 (code {result.returncode})")

    print(f"\n[{desc}] OK ({elapsed:.1f}s)")
    return result


def is_audio(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTS


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def build_final_path(input_path: Path, tgt_lang: str | None, output: Path | None) -> Path:
    if output is not None:
        return output

    if tgt_lang:
        # safe_lang = tgt_lang.strip().replace(" ", "_")
        return input_path.with_stem(f"{input_path.stem}").with_suffix(".srt")
    return input_path.with_suffix(".srt")


def main():
    parser = argparse.ArgumentParser(
        description="视频字幕生成 + 翻译一条龙",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认：自动生成中文字幕
  python generate_and_translate.py "input.mkv"

  # 生成英文字幕
  python generate_and_translate.py "input.mkv" --tgt-lang English

  # 已知源语言为日语，翻译为中文
  python generate_and_translate.py "input.mkv" --src-lang ja --tgt-lang 中文

  # 指定 ASR 和翻译设备
  python generate_and_translate.py "input.mkv" --asr-device CPU --trans-device GPU.1

  # 跳过翻译，只生成原语言字幕
  python generate_and_translate.py "input.mkv" --tgt-lang None

  # 保留中间原语言字幕
  python generate_and_translate.py "input.mkv" --keep-temp
        """,
    )

    parser.add_argument("input", help="输入视频或音频文件路径")

    # ASR options
    parser.add_argument("--src-lang", default="auto",
                        help="源语言 (默认: auto，自动检测；也可指定 en/zh/ja/ko 等)")
    parser.add_argument("--asr-device", default="GPU.1",
                        help="Whisper ASR 推理设备 (默认: GPU.1)")
    parser.add_argument("--asr-model-dir", default=DEFAULT_ASR_MODEL_DIR,
                        help="Whisper OpenVINO 模型目录 (默认: " + str(DEFAULT_ASR_MODEL_DIR) + ")")
    parser.add_argument("--whisper-timestamps", action="store_true",
                        help="使用 Whisper 内部 chunk 时间戳")
    parser.add_argument("--beam-search", action="store_true",
                        help="Whisper 启用 beam search")
    parser.add_argument("--num-beams", type=int, default=5,
                        help="beam search 数量 (默认: 5)")

    # Translation options
    parser.add_argument("--tgt-lang", default="中文",
                        help="目标语言 (默认: 中文；设为 None 跳过翻译)")
    parser.add_argument("--trans-device", default="GPU",
                        help="翻译模型推理设备 (默认: GPU)")
    parser.add_argument("--trans-model-dir", default=DEFAULT_TRANS_MODEL_DIR,
                        help="Hy-MT2 OpenVINO 模型目录 (默认: " + str(DEFAULT_TRANS_MODEL_DIR) + ")")
    parser.add_argument("--merge-chars", type=int, default=200,
                        help="翻译时合并短字幕的字符数 (默认: 200，0=逐条翻译)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="翻译最大生成长度 (默认: 512)")

    # Output options
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="最终 SRT 输出路径 (默认: 输入文件名_<目标语言>.srt)")
    parser.add_argument("--temp-dir", type=Path, default=None,
                        help="中间文件目录 (默认: 系统临时目录)")
    parser.add_argument("--keep-temp", action="store_true",
                        help="保留中间原语言字幕文件")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在: {input_path}")
        sys.exit(1)

    if not is_video(input_path) and not is_audio(input_path):
        print(f"[WARN] 无法识别文件类型: {input_path.suffix}，将尝试按视频处理")

    # Prepare the intermediate directory
    if args.temp_dir:
        temp_dir = args.temp_dir.absolute()
        temp_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="sub_translate_"))

    skip_translate = args.tgt_lang is None or args.tgt_lang.strip().lower() in ("none", "null", "")

    src_srt = temp_dir / f"{input_path.stem}_src.srt"
    final_srt = build_final_path(input_path, None if skip_translate else args.tgt_lang, args.output)

    total_start = time.time()

    try:
        # ========== Step 1: generate source-language subtitles ==========
        cmd1 = [
            sys.executable,
            str(Path(__file__).parent / "video_to_subtitle.py"),
            str(input_path),
            "--model-dir", args.asr_model_dir,
            "--device", args.asr_device,
            "--task", "transcribe",
            "--language", args.src_lang,
            "-o", str(src_srt),
        ]

        if is_audio(input_path):
            cmd1.append("--audio-only")
        if args.whisper_timestamps:
            cmd1.append("--whisper-timestamps")
        if args.beam_search:
            cmd1.extend(["--beam-search", "--num-beams", str(args.num_beams)])

        run_step(cmd1, "Step 1/2: 生成原语言字幕")

        if not src_srt.exists():
            raise RuntimeError(f"原语言字幕未生成: {src_srt}")

        # ========== Step 2: translate subtitles (optional) ==========
        if skip_translate:
            print("\n[Step 2/2] 跳过翻译，复制原语言字幕到最终路径")
            shutil.copy2(src_srt, final_srt)
        else:
            cmd2 = [
                sys.executable,
                str(Path(__file__).parent / "translate_srt.py"),
                str(src_srt),
                "--tgt-lang", args.tgt_lang,
                "--model-dir", args.trans_model_dir,
                "--device", args.trans_device,
                "--merge-chars", str(args.merge_chars),
                "--max-new-tokens", str(args.max_new_tokens),
                "-o", str(final_srt),
            ]
            run_step(cmd2, f"Step 2/2: 翻译为 {args.tgt_lang}")

        total_time = time.time() - total_start

        print("\n" + "=" * 60)
        print("[全部完成]")
        print("=" * 60)
        print(f"  输入: {input_path}")
        print(f"  输出: {final_srt.absolute()}")
        print(f"  总耗时: {total_time:.1f}s ({total_time / 60:.1f} min)")
        if not skip_translate:
            print(f"  中间字幕: {src_srt}")
        print("=" * 60)

    finally:
        # Clean up intermediate files
        if not args.keep_temp:
            try:
                if src_srt.exists():
                    src_srt.unlink()
                if args.temp_dir is None and temp_dir.exists():
                    temp_dir.rmdir()
            except Exception as e:
                print(f"[WARN] 清理临时文件失败: {e}")
        else:
            print(f"[INFO] 保留中间文件: {src_srt}")


if __name__ == "__main__":
    main()
