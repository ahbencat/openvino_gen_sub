#!/usr/bin/env python3
"""
Batch video subtitle generation + translation.

Usage:
  python batch_generate.py "./videos"
  python batch_generate.py "./videos" --src-lang ja --tgt-lang English
  python batch_generate.py --dry-run    # only list files, do not process
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent

# Default video directory: current directory
DEFAULT_VIDEO_DIR = Path(".")

# Default target script
TARGET_SCRIPT = SCRIPT_DIR / "generate_and_translate.py"

# Video extensions to process
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v"}


def find_python() -> str:
    """Locate Python interpreter: --python arg > local venv > current interpreter."""
    for candidate in (SCRIPT_DIR / ".venv" / "Scripts" / "python.exe",
                      SCRIPT_DIR / ".venv" / "bin" / "python"):
        if candidate.exists():
            return str(candidate)

    return sys.executable


def collect_videos(video_dir: Path, recursive: bool = True) -> list[Path]:
    """Collect video files in directory."""
    if not video_dir.exists():
        print(f"[ERROR] Directory not found: {video_dir}")
        return []

    pattern_files = []
    if recursive:
        for ext in VIDEO_EXTS:
            pattern_files.extend(video_dir.rglob(f"*{ext}"))
            pattern_files.extend(video_dir.rglob(f"*{ext.upper()}"))
    else:
        for ext in VIDEO_EXTS:
            pattern_files.extend(video_dir.glob(f"*{ext}"))
            pattern_files.extend(video_dir.glob(f"*{ext.upper()}"))

    # Deduplicate and sort
    seen = set()
    unique = []
    for p in pattern_files:
        if p not in seen and p.is_file():
            seen.add(p)
            unique.append(p)
    unique.sort()
    return unique


def run_one(python: str, video: Path, extra_args: list[str]) -> bool:
    """Run generate_and_translate.py on a single video. Returns True on success."""
    cmd = [python, str(TARGET_SCRIPT), str(video), *extra_args]

    print()
    print("=" * 70)
    print(f"Processing: {video.name}")
    print(f"Path:      {video}")
    print(f"Command:   {' '.join(cmd)}")
    print("=" * 70)
    print()

    start = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    except KeyboardInterrupt:
        print("\n[INTERRUPT] User aborted.")
        return False
    except Exception as e:
        print(f"[EXCEPTION] {e}")
        return False

    elapsed = time.time() - start
    ok = result.returncode == 0
    status = "OK" if ok else "FAIL"
    print(f"\n[{status}] {video.name} ({elapsed:.1f}s)")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Batch video subtitle generation + translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_generate.py
  python batch_generate.py "./videos"
  python batch_generate.py --src-lang ja --tgt-lang 中文
  python batch_generate.py --dry-run
  python batch_generate.py --no-recursive
  python batch_generate.py --skip-existing   # skip if .srt already exists
        """,
    )

    parser.add_argument(
        "video_dir",
        nargs="?",
        default=str(DEFAULT_VIDEO_DIR),
        help="Video directory to scan (default: current directory)",
    )
    parser.add_argument(
        "--src-lang",
        default="auto",
        help="Source language passed to generate_and_translate.py (default: auto)",
    )
    parser.add_argument(
        "--tgt-lang",
        default="中文",
        help="Target language passed to generate_and_translate.py (default: 中文)",
    )
    parser.add_argument(
        "--asr-device",
        default="GPU.1",
        help="Whisper ASR device (default: GPU.1)",
    )
    parser.add_argument(
        "--trans-device",
        default="GPU.1",
        help="Translation model device (default: GPU.1)",
    )
    parser.add_argument(
        "--whisper-timestamps",
        action="store_true",
        default=True,
        help="Use Whisper internal chunk timestamps (default: ON)",
    )
    parser.add_argument(
        "--beam-search",
        action="store_true",
        default=True,
        help="Enable Whisper beam search (default: ON)",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=5,
        help="Beam count (default: 5)",
    )
    parser.add_argument(
        "--merge-chars",
        type=int,
        default=200,
        help="Max chars per translation block (default: 200)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan top-level directory",
    )
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Skip video if the output .srt already exists (default: ON)",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Disable skipping, re-generate subtitles even if .srt exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list files, do not process",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python interpreter path (overrides auto-detection)",
    )

    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    python = args.python or find_python()

    print("=" * 70)
    print("Batch Subtitle Generation + Translation")
    print("=" * 70)
    print(f"  Dir:        {video_dir}")
    print(f"  Python:     {python}")
    print(f"  Script:     {TARGET_SCRIPT}")
    print(f"  Recursive:  {not args.no_recursive}")
    print(f"  Skip exist: {args.skip_existing}")
    print(f"  Dry run:    {args.dry_run}")
    print(f"  Args:       src={args.src_lang} tgt={args.tgt_lang} "
          f"asr={args.asr_device} trans={args.trans_device} "
          f"timestamps={args.whisper_timestamps} beam={args.beam_search}")
    print("=" * 70)

    if not TARGET_SCRIPT.exists():
        print(f"\n[FATAL] Target script not found: {TARGET_SCRIPT}")
        sys.exit(1)

    videos = collect_videos(video_dir, recursive=not args.no_recursive)
    print(f"\nFound {len(videos)} video file(s).")

    if not videos:
        print("[DONE] Nothing to process.")
        return

    if args.dry_run:
        print("\nDry run - listing files:")
        for i, v in enumerate(videos, 1):
            print(f"  {i:4d}. {v}")
        return

    # Build the extra args passed to generate_and_translate.py
    extra_args = [
        "--src-lang", args.src_lang,
        "--tgt-lang", args.tgt_lang,
        "--asr-device", args.asr_device,
        "--trans-device", args.trans_device,
        "--merge-chars", str(args.merge_chars),
    ]
    if args.whisper_timestamps:
        extra_args.append("--whisper-timestamps")
    if args.beam_search:
        extra_args.append("--beam-search")
        extra_args.extend(["--num-beams", str(args.num_beams)])

    total = len(videos)
    success = 0
    failed = 0
    skipped = 0
    total_start = time.time()

    for i, video in enumerate(videos, 1):
        print(f"\n>>> [{i}/{total}]")

        if args.skip_existing:
            # Output filename per generate_and_translate.py build_final_path():
            # {stem}.srt (same name as video, regardless of tgt-lang)
            candidate = video.with_suffix(".srt")
            if candidate.exists():
                print(f"[SKIP] Output already exists: {candidate}")
                skipped += 1
                continue

        if run_one(python, video, extra_args):
            success += 1
        else:
            failed += 1

    total_elapsed = time.time() - total_start
    print()
    print("=" * 70)
    print("Batch processing done")
    print("=" * 70)
    print(f"  Total:   {total}")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Elapsed: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
