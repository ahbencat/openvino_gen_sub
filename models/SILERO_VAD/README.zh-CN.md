# Silero VAD

[English](README.md) | **简体中文**

[Silero VAD](https://github.com/snakers4/silero-vad) —— 轻量预训练语音活动检测模型，ONNX 格式。`video_to_subtitle.py` 转录前用它切分语音段。

- **文件**：`silero_vad.onnx`（约 2.3 MB，v5）
- **来源**：[snakers4/silero-vad](https://github.com/snakers4/silero-vad)（`src/silero_vad/data/silero_vad.onnx`），也随 [`silero-vad`](https://pypi.org/project/silero-vad/) pip 包附带
- **许可**：MIT

## 脚本从哪里找它

`video_to_subtitle.py` 以自身位置为基准解析路径：

```
<仓库根>/models/SILERO_VAD/silero_vad.onnx
```

文件缺失时脚本启动即报 `FileNotFoundError`，并给出预期路径。从上面来源重新下载，或装 pip 包后把附带的 onnx 拷过来即可恢复：

```bash
# 方式一：直接下载
curl -L -o models/SILERO_VAD/silero_vad.onnx \
    https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

# 方式二：从 pip 包提取
pip install silero-vad
python -c "import silero_vad, shutil; print(silero_vad.__file__)"
# 然后把该包目录下的 silero_vad.onnx 复制到 models/SILERO_VAD/
```

## 在本项目中的用法

每次运行加载一次（`load_silero_vad`），输入 16 kHz 单声道音频；`get_speech_timestamps` 返回的语音起止点即字幕分段依据。可调参数：`--vad-threshold`、`--vad-merge-gap`、`--vad-padding`；`--no-vad` 完全跳过 VAD（固定 30 秒分块）。
