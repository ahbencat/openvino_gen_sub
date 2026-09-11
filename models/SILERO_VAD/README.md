# Silero VAD

**English** | [简体中文](README.zh-CN.md)

[Silero VAD](https://github.com/snakers4/silero-vad) — a compact, pre-trained voice activity detector — in ONNX format. This is the model that `video_to_subtitle.py` uses to find speech segments before transcription.

- **File**: `silero_vad.onnx` (~2.3 MB, v5)
- **Source**: [snakers4/silero-vad](https://github.com/snakers4/silero-vad) (`src/silero_vad/data/silero_vad.onnx`), also bundled with the [`silero-vad`](https://pypi.org/project/silero-vad/) pip package
- **License**: MIT

## Where the scripts look for it

`video_to_subtitle.py` resolves the path relative to its own location:

```
<repo root>/models/SILERO_VAD/silero_vad.onnx
```

If the file is missing, the script fails on startup with a `FileNotFoundError` naming the expected path. Restore it either by downloading from the source above or by reinstalling the pip package and copying the bundled onnx:

```bash
# Option 1: direct download
curl -L -o models/SILERO_VAD/silero_vad.onnx \
    https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

# Option 2: from the pip package
pip install silero-vad
python -c "import silero_vad, shutil; print(silero_vad.__file__)"
# then copy silero_vad.onnx from that package directory to models/SILERO_VAD/
```

## Usage in this project

The model is loaded once per run (`load_silero_vad`) and fed 16 kHz mono audio; `get_speech_timestamps` returns speech boundaries which become subtitle segment timings. Tuning knobs: `--vad-threshold`, `--vad-merge-gap`, `--vad-padding`; `--no-vad` skips VAD entirely (fixed 30 s chunks).
