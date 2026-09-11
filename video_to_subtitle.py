#!/usr/bin/env python3
"""
Video subtitle generator based on OpenVINO + Whisper Large-v3.

Usage:
    python video_to_subtitle.py "input.mkv"
    python video_to_subtitle.py "input.mkv" --device GPU.1 --language en --task translate
"""

import os
import sys
import argparse
import subprocess
import tempfile
import json
import time
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import transformers
import math

transformers.logging.set_verbosity_error()

# Try to import the OpenVINO-related libraries
try:
    from optimum.intel import OVModelForSpeechSeq2Seq
    from transformers import AutoProcessor, pipeline
except ImportError as e:
    print("缺少依赖:", e)
    print("请运行: pip install optimum-intel[openvino] transformers")
    sys.exit(1)


# ============= Config =============
SCRIPT_DIR = Path(__file__).parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "models" / "whisper-large-v3-fp16-ov"
DEFAULT_DEVICE = "GPU.1"
DEFAULT_TASK = "translate"
DEFAULT_LANGUAGE = "en"
# ffmpeg executable: resolved from PATH by default,
# or set the FFMPEG_PATH environment variable to a full path
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")


# ============= Audio extraction =============
def extract_audio(video_path, output_wav=None, sample_rate=16000):
    """Extract audio from a video as 16 kHz mono wav."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError("视频文件不存在: " + str(video_path))

    if output_wav is None:
        output_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    output_wav = Path(output_wav)

    print("\n[extract_audio]", video_path.name)

    # Extract audio with ffmpeg, compensating the MP4/AAC timeline into the PCM WAV
    cmd = [
        FFMPEG_PATH,
        "-copyts",
        "-start_at_zero",
        "-i", str(video_path),
        "-map", "0:a:0",
        "-vn",
        "-af", "aresample=async=1:first_pts=0",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-f", "wav",
        "-y",
        str(output_wav)
    ]

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        raise RuntimeError(
            "找不到 ffmpeg。请先安装 ffmpeg 并加入 PATH，"
            "或设置环境变量 FFMPEG_PATH 指向 ffmpeg 可执行文件"
        ) from None
    elapsed = time.time() - start

    if result.returncode != 0:
        print("ffmpeg 错误:\n", result.stderr[-1000:])
        raise RuntimeError("音频提取失败")

    if not output_wav.exists() or output_wav.stat().st_size == 0:
        raise RuntimeError("生成的音频文件为空")

    size_mb = output_wav.stat().st_size / (1024 * 1024)
    print("OK", str(output_wav.name), "({:.1f} MB, {:.1f}s)".format(size_mb, elapsed))

    return str(output_wav)


# ============= SRT formatting =============
def format_timestamp(seconds, default=0):
    """Seconds -> SRT timestamp format: 00:00:00,000."""
    if seconds is None:
        seconds = default
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return "{:02d}:{:02d}:{:02d},{:03d}".format(hours, minutes, secs, millis)


def write_srt(chunks, output_path):
    """
    Write timestamped chunks to an SRT file.

    chunks format:
    [
        {"timestamp": (start, end), "text": "..."},
        ...
    ]
    """
    output_path = Path(output_path)
    print("\n[SRT write]", output_path)

    entries = []
    SUBTITLE_GAP = 0.1

    for chunk in chunks:
        start, end = chunk["timestamp"]
        text = chunk["text"].strip()

        if not text:
            continue

        if entries and entries[-1][1] >= start:
            prev_start, prev_end, prev_text = entries[-1]
            entries[-1] = (prev_start, max(prev_start + 0.1, start - SUBTITLE_GAP), prev_text)
        if end <= start:
            end = start + 1.0

        entries.append((start, end, text))

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, (start, end, text) in enumerate(entries, 1):
            f.write(str(idx) + "\n")
            f.write(format_timestamp(start) + " --> " + format_timestamp(end) + "\n")
            f.write(text + "\n\n")

    print("OK", len(entries), "segments")
    return str(output_path)


# ============= Silero VAD =============
VAD_MODEL_PATH = Path(__file__).parent / "models" / "SILERO_VAD" / "silero_vad.onnx"

def load_vad_model():
    """Load the Silero VAD model (local ONNX file, global singleton)."""
    if not hasattr(load_vad_model, "model"):
        if not VAD_MODEL_PATH.exists():
            raise FileNotFoundError("VAD 模型文件未找到: " + str(VAD_MODEL_PATH))
        print("\n[load_vad]", VAD_MODEL_PATH.name)
        start = time.time()
        from silero_vad import load_silero_vad, get_speech_timestamps
        load_vad_model.model = load_silero_vad(str(VAD_MODEL_PATH))
        load_vad_model.get_speech_timestamps = get_speech_timestamps
        print("OK ({:.1f}s)".format(time.time() - start))
    return load_vad_model.model, load_vad_model.get_speech_timestamps


def vad_get_segments(audio_path, model, get_speech_timestamps,
                     threshold=0.6, min_speech_duration_ms=250,
                     min_silence_duration_ms=100, merge_gap_ms=500,
                     padding_ms=300):
    """Detect speech segments with VAD; returns (segments, audio, sr).

    segments: [(padded_start, padded_end, speech_start), ...]
    speech_start is the actual speech onset with the left padding removed,
    used to correct timestamps.
    """
    audio, sr = sf.read(audio_path, dtype='float32')

    # Downmix multi-channel audio to mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Silero VAD requires 8 kHz or 16 kHz audio
    if sr != 16000:
        print("WARN: VAD works best at 16kHz, got {}Hz".format(sr))

    raw_segments = get_speech_timestamps(
        audio, model, sampling_rate=sr,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        return_seconds=True
    )

    duration = len(audio) / sr

    merged_speech = []
    for seg in raw_segments:
        speech_start = seg['start']
        speech_end = seg['end']
        if merged_speech and speech_start - merged_speech[-1][1] < merge_gap_ms / 1000:
            merged_speech[-1] = (merged_speech[-1][0], max(merged_speech[-1][1], speech_end))
        else:
            merged_speech.append((speech_start, speech_end))

    padded = []
    for speech_start, speech_end in merged_speech:
        start = max(0, speech_start - padding_ms / 1000)
        end = min(duration, speech_end + padding_ms / 1000)
        padded.append((start, end, speech_start, speech_end))

    return padded, audio, sr, raw_segments


# ============= Main processing class =============
class VideoSubtitleGenerator:
    def __init__(self, model_dir=None, device=DEFAULT_DEVICE, cache_dir="./model_cache"):
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self.device = device
        self.cache_dir = Path(cache_dir).absolute()
        self.cache_dir.mkdir(exist_ok=True)

        self.model = None
        self.processor = None
        self.pipe = None

    def load_model(self):
        """Load the model (slow, happens only once)."""
        print("\n[load_model]", Path(self.model_dir).name)
        print("   device:", self.device)

        start = time.time()

        self.model = OVModelForSpeechSeq2Seq.from_pretrained(
            self.model_dir,
            device=self.device,
            ov_config={"CACHE_DIR": str(self.cache_dir)}
        )

        self.processor = AutoProcessor.from_pretrained(self.model_dir)

        load_time = time.time() - start
        print("OK ({:.1f}s)".format(load_time))

        return self

    def build_pipeline(self, chunk_length_s=30, batch_size=1):
        """Build the ASR pipeline."""
        if not self.model or not self.processor:
            raise RuntimeError("请先调用 load_model()")

        print("\n[build_pipeline] chunk={}s, batch={}".format(chunk_length_s, batch_size))
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            return_timestamps=True,
            framework="pt",
        )
        print("OK")
        return self

    @staticmethod
    def _print_timing_report(timings, label="chunk"):
        """Print a timing statistics report: avg / P90 / P95."""
        if not timings:
            return
        arr = sorted(timings)
        n = len(arr)
        avg = sum(arr) / n
        p90 = arr[int(n * 0.9)]
        p95 = arr[int(n * 0.95)]
        print("\n--- Timing Report ({}, n={}) ---".format(label, n))
        print("   avg: {:.2f}s".format(avg))
        print("   P90: {:.2f}s".format(p90))
        print("   P95: {:.2f}s".format(p95))

    @staticmethod
    def _gen_kwargs(task, language, num_beams=None):
        kwargs = {"task": task}
        if language and language != "auto":
            kwargs["language"] = language
        if num_beams is not None and num_beams > 1:
            kwargs["num_beams"] = num_beams
            kwargs["early_stopping"] = True
        return kwargs

    @staticmethod
    def _split_long_segments(segments, max_duration=15.0):
        """Evenly split VAD segments longer than max_duration seconds into sub-segments."""
        result = []
        for seg_start, seg_end, speech_start, speech_end in segments:
            duration = seg_end - seg_start
            if duration <= max_duration:
                result.append((seg_start, seg_end, speech_start, speech_end))
            else:
                n = int(math.ceil(duration / max_duration))
                chunk_len = duration / n
                for i in range(n):
                    s = seg_start + i * chunk_len
                    e = seg_start + (i + 1) * chunk_len if i < n - 1 else seg_end
                    result.append((s, e, max(s, speech_start), min(e, speech_end)))
        return result

    def transcribe(self, audio_path, task=DEFAULT_TASK, language=DEFAULT_LANGUAGE, num_beams=None):
        """Transcribe audio."""
        if not self.pipe:
            self.build_pipeline()

        print("\n[transcribe]", Path(audio_path).name)
        print("   task:", task, "| language:", language)
        gen_kwargs = self._gen_kwargs(task, language, num_beams)
        print("   generate_kwargs:", gen_kwargs)

        start = time.time()

        result = self.pipe(audio_path, generate_kwargs=gen_kwargs)

        elapsed = time.time() - start

        print("OK ({:.1f}s)".format(elapsed))
        print("   text length:", len(result["text"]))
        print("   segments:", len(result.get("chunks", [])))

        return result

    def transcribe_with_vad(self, audio_path, task=DEFAULT_TASK, language=DEFAULT_LANGUAGE,
                            vad_threshold=0.5, vad_merge_gap_ms=500, vad_padding_ms=300,
                            chunk_length_s=30, batch_size=1, num_beams=None,
                            whisper_timestamps=False,
                            whisper_start_offset=0.0, whisper_end_offset=0.0):
        """Detect speech segments with VAD, then transcribe segment by segment."""
        if not self.pipe:
            self.build_pipeline(chunk_length_s=chunk_length_s, batch_size=batch_size)

        print("\n[transcribe + VAD]", Path(audio_path).name)
        print("   task:", task, "| language:", language)
        print("   batch_size:", batch_size)
        timestamps_mode = "Whisper 内部时间戳" if whisper_timestamps else "VAD 段边界"
        print(f"   时间戳模式: {timestamps_mode}")
        gen_kwargs = self._gen_kwargs(task, language, num_beams)
        print("   generate_kwargs:", gen_kwargs)

        total_start = time.time()

        # 1. Load the VAD model and detect speech segments
        vad_model, get_speech_timestamps = load_vad_model()
        vad_start = time.time()
        segments, audio, sr, raw_segments = vad_get_segments(
            audio_path, vad_model, get_speech_timestamps,
            merge_gap_ms=vad_merge_gap_ms, padding_ms=vad_padding_ms
        )
        n_merged = len(segments)
        if whisper_timestamps:
            # Whisper internal timestamps mode: keep long segments for more context
            print("   VAD: {} segments (merged from {}) [{:.1f}s]".format(
                len(segments), len(raw_segments), time.time() - vad_start))
        else:
            segments = self._split_long_segments(segments, max_duration=15.0)
            print("   VAD: {} segments (merged from {}, split to {}) [{:.1f}s]".format(
                len(segments), n_merged, len(raw_segments), time.time() - vad_start))

        if not segments:
            print("WARN: no speech detected by VAD")
            return {"text": "", "chunks": []}

        total_segments = len(segments)
        print("   transcribe segments: {}\n".format(total_segments))

        # 2. Transcribe segment by segment
        all_chunks = []
        full_text_parts = []
        segment_timings = []

        for idx, (seg_start, seg_end, speech_start, speech_end) in enumerate(segments):
            seg_len = seg_end - seg_start
            print("\r   [{}/{}] {:4.1f}s".format(idx + 1, total_segments, seg_len), end="", flush=True)
            if seg_len < 0.3:
                print(" (skip)")
                continue

            start_sample = int(seg_start * sr)
            end_sample = int(seg_end * sr)
            seg_audio = audio[start_sample:end_sample]

            t0 = time.time()
            seg_result = self.pipe({"array": seg_audio, "sampling_rate": int(sr)}, generate_kwargs=gen_kwargs)
            seg_time = time.time() - t0
            segment_timings.append(seg_time)

            seg_text = seg_result["text"].strip()
            if not seg_text:
                continue

            full_text_parts.append(seg_text)

            if whisper_timestamps:
                # Use Whisper's internal chunk timestamps
                chunks = seg_result.get("chunks", [])
                if chunks:
                    for chunk in chunks:
                        raw_start, raw_end = chunk.get("timestamp", (None, None))
                        if raw_start is None:
                            raw_start = 0
                        if raw_end is None:
                            raw_end = raw_start + 5
                        abs_start = seg_start + raw_start
                        abs_end = seg_start + raw_end
                        abs_start = max(speech_start, abs_start + whisper_start_offset)
                        abs_end = min(speech_end, abs_end + whisper_end_offset)
                        abs_end = max(abs_start + 0.1, abs_end)
                        all_chunks.append({
                            "timestamp": (abs_start, abs_end),
                            "text": chunk["text"],
                        })
                    continue
            # Fallback / VAD boundary mode: use the VAD segment boundaries
            if speech_end <= speech_start:
                continue
            all_chunks.append({
                "timestamp": (speech_start, speech_end),
                "text": seg_text
            })

        elapsed = time.time() - total_start

        result = {
            "text": " ".join(full_text_parts),
            "chunks": all_chunks,
            "segment_timings": segment_timings,
        }

        self._print_timing_report(segment_timings, label="segment")

        print("\nOK ({:.1f}s)".format(elapsed))
        print("   text length:", len(result["text"]))
        print("   segments:", len(all_chunks))

        return result

    def process_video(self, video_path, output_path=None, keep_audio=False, use_vad=False,
                      vad_threshold=0.5, vad_merge_gap_ms=500, vad_padding_ms=300,
                      chunk_length_s=30, batch_size=1,
                      **kwargs):
        """Full video processing pipeline."""
        video_path = Path(video_path)
        print("=" * 60)
        print("[VIDEO]", video_path.name)
        print("=" * 60)

        total_start = time.time()

        # 1. Extract audio
        if keep_audio:
            temp_audio = video_path.with_suffix(".wav")
        else:
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

        try:
            audio_path = extract_audio(video_path, temp_audio)

            # 2. Transcribe
            if use_vad:
                result = self.transcribe_with_vad(
                    audio_path,
                    vad_threshold=vad_threshold,
                    vad_merge_gap_ms=vad_merge_gap_ms,
                    vad_padding_ms=vad_padding_ms,
                    chunk_length_s=chunk_length_s,
                    batch_size=batch_size,
                    **kwargs
                )
            else:
                result = self.transcribe(audio_path, **kwargs)

            # 3. Write the SRT
            if output_path is None:
                output_path = video_path.with_suffix(".srt")

            if "chunks" in result and result["chunks"]:
                srt_path = write_srt(result["chunks"], output_path)
            else:
                print("WARN: no chunk timestamps, writing full text")
                srt_path = write_srt([{"timestamp": (0, 0), "text": result["text"]}], output_path)

            total_time = time.time() - total_start

            print("\n" + "=" * 60)
            if "segment_timings" in result and result["segment_timings"]:
                self._print_timing_report(result["segment_timings"], label="segment")
            print("\n[DONE] total time: {:.1f}s".format(total_time))
            print("   SRT:", Path(srt_path).absolute())
            print("   text length:", len(result["text"]))
            print("=" * 60)

            return {
                "video": str(video_path),
                "srt": srt_path,
                "text": result["text"],
                "chunks": result.get("chunks", []),
                "total_time": total_time,
                "segment_timings": result.get("segment_timings", []),
            }

        finally:
            if not keep_audio:
                Path(temp_audio).unlink(missing_ok=True)


# ============= CLI =============
def main():
    parser = argparse.ArgumentParser(
        description="视频字幕生成工具 (OpenVINO + Whisper Large-v3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 最简单用法
  python video_to_subtitle.py "F:/Video/Movie/Ghostbusters...mkv"

  # 自定义参数
  python video_to_subtitle.py "input.mkv" --device GPU.1 --task translate --language en

  # 保留中间音频文件 (调试)
  python video_to_subtitle.py "input.mkv" --keep-audio

  # 调试单音频文件 (不提取音轨)
  python video_to_subtitle.py "audio.wav" --audio-only
        """
    )

    parser.add_argument("input", help="输入视频或音频文件路径")

    # Model options
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help="模型目录 (默认: " + str(DEFAULT_MODEL_DIR) + ")")
    parser.add_argument("--device", default=DEFAULT_DEVICE,
                        help="推理设备: CPU | GPU.0 | GPU.1 | NPU (默认: " + DEFAULT_DEVICE + ")")

    # Transcription options
    parser.add_argument("--task", default=DEFAULT_TASK,
                        choices=["transcribe", "translate"],
                        help="任务: transcribe(原样转录) | translate(翻译为英文, Whisper仅支持→en) (默认: " + DEFAULT_TASK + ")")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE,
                        help="源语言 (默认: " + DEFAULT_LANGUAGE + ", auto=自动检测)")
    parser.add_argument("--chunk-length", type=int, default=30,
                        help="音频分块长度秒 (默认: 30, Whisper原生块大小)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="推理批大小 (默认: 1, 不建议超过2)")
    parser.add_argument("--beam-search", action="store_true", dest="beam_search",
                        help="启用 beam search (默认禁用, 使用贪心搜索)")
    parser.add_argument("--num-beams", type=int, default=5,
                        help="beam search 的 beam 数量 (默认: 5, 仅 --beam-search 启用时生效)")
    parser.add_argument("--whisper-timestamps", action="store_true", dest="whisper_timestamps",
                        help="使用 Whisper 内部 chunk 时间戳 (默认: 使用 VAD 段边界)")
    parser.add_argument("--whisper-start-offset", type=float, default=0.0,
                        help="Whisper chunk 起始时间偏移秒，仅 --whisper-timestamps 生效，可为负数")
    parser.add_argument("--whisper-end-offset", type=float, default=0.0,
                        help="Whisper chunk 结束时间偏移秒，仅 --whisper-timestamps 生效，可为负数")

    # Output options
    parser.add_argument("--output", "-o", help="SRT输出路径 (默认: 视频同目录同名.srt)")
    parser.add_argument("--output-json", help="同时导出带时间戳的 JSON 文件")
    parser.add_argument("--output-txt", help="同时导出纯文本文件")

    # Debug options
    parser.add_argument("--keep-audio", action="store_true",
                        help="保留中间 wav 音频文件 (调试用)")
    parser.add_argument("--audio-only", action="store_true",
                        help="输入是音频文件, 跳过 ffmpeg 提取")
    parser.add_argument("--cache-dir", default="./model_cache",
                        help="OpenVINO 模型缓存目录 (默认: ./model_cache)")

    # VAD options
    parser.add_argument("--vad", action="store_true", dest="vad", default=True,
                        help=argparse.SUPPRESS)
    parser.add_argument("--no-vad", action="store_false", dest="vad",
                        help="禁用 VAD (默认启用, 改为固定30s分块)")
    parser.add_argument("--vad-threshold", type=float, default=0.5,
                        help="VAD 判定阈值 (默认: 0.5)")
    parser.add_argument("--vad-merge-gap", type=int, default=500,
                        help="VAD 合并间隙毫秒 (默认: 500)")
    parser.add_argument("--vad-padding", type=int, default=300,
                        help="VAD 段前后填充毫秒 (默认: 300)")

    args = parser.parse_args()

    # Print the start banner
    print("\n" + "=" * 60)
    print("OPENVINO WHISPER SUBTITLE GENERATOR")
    print("=" * 60)

    # Create the generator
    generator = VideoSubtitleGenerator(
        model_dir=args.model_dir,
        device=args.device,
        cache_dir=args.cache_dir
    )

    # num_beams: only takes effect when --beam-search is set
    num_beams = args.num_beams if args.beam_search else None

    # Load the model
    generator.load_model()
    generator.build_pipeline(chunk_length_s=args.chunk_length, batch_size=args.batch_size)

    # Process
    if args.audio_only:
        # Audio file: process directly
        if args.vad:
            result = generator.transcribe_with_vad(
                args.input, task=args.task, language=args.language,
                vad_threshold=args.vad_threshold,
                vad_merge_gap_ms=args.vad_merge_gap,
                vad_padding_ms=args.vad_padding,
                chunk_length_s=args.chunk_length,
                batch_size=args.batch_size,
                num_beams=num_beams,
                whisper_timestamps=args.whisper_timestamps,
                whisper_start_offset=args.whisper_start_offset,
                whisper_end_offset=args.whisper_end_offset,
            )
        else:
            result = generator.transcribe(args.input, task=args.task, language=args.language, num_beams=num_beams)
        if args.output:
            srt_path = write_srt(result.get("chunks", [{"timestamp": (0, 0), "text": result["text"]}]), args.output)
            print("SRT saved:", srt_path)
    else:
        # Video: full pipeline
        result = generator.process_video(
            args.input,
            output_path=args.output,
            keep_audio=args.keep_audio,
            use_vad=args.vad,
            vad_threshold=args.vad_threshold,
            vad_merge_gap_ms=args.vad_merge_gap,
            vad_padding_ms=args.vad_padding,
            chunk_length_s=args.chunk_length,
            batch_size=args.batch_size,
            task=args.task,
            language=args.language,
            num_beams=num_beams,
            whisper_timestamps=args.whisper_timestamps,
            whisper_start_offset=args.whisper_start_offset,
            whisper_end_offset=args.whisper_end_offset,
        )

    # Extra outputs
    if args.output_json and "chunks" in result:
        json_path = Path(args.output_json)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result["chunks"], f, ensure_ascii=False, indent=2)
        print("JSON saved:", json_path)

    if args.output_txt:
        txt_path = Path(args.output_txt)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print("TXT saved:", txt_path)

    print("\nAll done!")


if __name__ == "__main__":
    main()
