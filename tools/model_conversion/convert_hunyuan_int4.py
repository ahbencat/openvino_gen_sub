#!/usr/bin/env python3
"""
Hy-MT2-7B -> OpenVINO INT4 (两步：先导出 FP16 + stateful，再 NNCF 权重量化)
"""

import argparse
import gc
import shutil
from pathlib import Path

import numpy as np
import openvino as ov
import tempfile
import torch


MODEL_DIR = Path(r"D:\Projects\whisper_cpp_win\Hy-MT2-7B")
OUTPUT_DIR = Path(r"D:\Projects\whisper_cpp_win\Hy-MT2-7B-ov-int4")


def register_hunyuan_config():
    from optimum.exporters.openvino.model_configs import Qwen2OpenVINOConfig
    from optimum.exporters.tasks import TasksManager

    register = TasksManager.create_register("openvino", overwrite_existing=True)

    @register(
        "hunyuan_v1_dense",
        "text-generation",
        "text-generation-with-past",
        library_name="transformers",
    )
    class HunyuanDenseV1OpenVINOConfig(Qwen2OpenVINOConfig):
        pass


def export_fp16(device: str):
    """Step 1: export stateful FP16 model."""
    ov_dir = OUTPUT_DIR / "_fp16"
    if ov_dir.exists():
        shutil.rmtree(ov_dir)

    register_hunyuan_config()

    from optimum.exporters.openvino import main_export
    from optimum.intel.openvino import OVConfig

    print("[1/2] Exporting FP16 stateful model...")
    main_export(
        model_name_or_path=str(MODEL_DIR),
        output=str(ov_dir),
        task="text-generation-with-past",
        device=device.lower(),
        framework="pt",
        trust_remote_code=True,
        library_name="transformers",
        stateful=True,
        local_files_only=True,
        model_loading_kwargs={
            "torch_dtype": torch.float16,
            "attn_implementation": "eager",
            "low_cpu_mem_usage": True,
        },
        ov_config=OVConfig(dtype="fp16"),
    )

    xml_path = ov_dir / "openvino_model.xml"
    bin_path = ov_dir / "openvino_model.bin"
    if not xml_path.exists():
        raise RuntimeError("FP16 export failed: openvino_model.xml not found")

    fp16_gb = bin_path.stat().st_size / 1024**3
    print(f"  FP16 model saved ({fp16_gb:.1f} GB)")
    return ov_dir


def quantize_int4(fp16_dir: Path):
    """Step 2: NNCF INT4 weight compression on the exported IR."""
    print()
    print("[2/2] Applying NNCF INT4 weight compression...")

    try:
        import nncf
    except ImportError:
        raise ImportError("nncf is required for INT4 quantization. Install with: pip install nncf")

    # Patch Rich's Windows renderer to avoid GBK crash with • character
    import rich._windows_renderer
    rich._windows_renderer.legacy_windows_render = lambda buffer, term: None

    model_path = fp16_dir / "openvino_model.xml"
    core = ov.Core()
    model = core.read_model(model_path)

    # INT4 weight-only compression
    compressed = nncf.compress_weights(
        model,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        group_size=128,
        ratio=0.9,
    )

    # Save to a fresh temp dir (avoids locked .bin in fp16_dir)
    tmp_dir = Path(tempfile.mkdtemp())
    ov.save_model(compressed, tmp_dir / "openvino_model.xml")
    int4_gb = (tmp_dir / "openvino_model.bin").stat().st_size / 1024**3
    print(f"  INT4 model saved ({int4_gb:.2f} GB)")

    # Release model handles
    del model, compressed
    gc.collect()

    # Copy tokenizer / config from fp16_dir -> tmp_dir
    for f in fp16_dir.iterdir():
        if f.suffix in (".json", ".jinja"):
            shutil.copy2(f, tmp_dir / f.name)

    # Replace OUTPUT_DIR with tmp_dir (use shutil.move for cross-drive)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    shutil.rmtree(fp16_dir, ignore_errors=True)
    shutil.move(str(tmp_dir), str(OUTPUT_DIR))

    return int4_gb


def show_output():
    print()
    print("[OK] Done!")
    print(f"  Output: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.rglob("*")):
        if f.is_file() and f.suffix in (".xml", ".bin"):
            mb = f.stat().st_size / 1024**2
            print(f"    {f.relative_to(OUTPUT_DIR)}  ({mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    fp16_dir = OUTPUT_DIR / "_fp16"
    if not fp16_dir.exists() or not (fp16_dir / "openvino_model.xml").exists():
        fp16_dir = export_fp16(args.device)
    else:
        print("[1/2] FP16 temp dir already exists, skipping export")

    int4_gb = quantize_int4(fp16_dir)
    show_output()


if __name__ == "__main__":
    main()