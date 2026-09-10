#!/usr/bin/env python3
"""
Hy-MT2 / Hunyuan Dense V1 -> OpenVINO via optimum-intel exporter.

This script registers the missing `hunyuan_v1_dense` OpenVINO export config locally,
then calls optimum-intel's official `main_export()` path.
"""

import argparse
import shutil
from pathlib import Path

import torch


def register_hunyuan_openvino_config():
    import optimum.exporters.openvino.model_configs
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

    return HunyuanDenseV1OpenVINOConfig


def convert(model_dir: Path, output_dir: Path, task: str, device: str, fp16: bool, clean: bool):
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    register_hunyuan_openvino_config()

    from optimum.exporters.openvino import main_export
    from optimum.intel.openvino import OVConfig

    print("=" * 60)
    print("Hy-MT2 / Hunyuan Dense V1 -> OpenVINO")
    print("=" * 60)
    print(f"  Model:  {model_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Task:   {task}")
    print(f"  Device: {device}")
    print(f"  FP16:   {fp16}")
    print()

    ov_config = OVConfig(dtype="fp16" if fp16 else "fp32")

    main_export(
        model_name_or_path=str(model_dir),
        output=str(output_dir),
        task=task,
        device=device.lower(),
        framework="pt",
        trust_remote_code=True,
        library_name="transformers",
        stateful=True,
        local_files_only=True,
        model_loading_kwargs={
            "torch_dtype": torch.float16 if fp16 else torch.float32,
            "attn_implementation": "eager",
            "low_cpu_mem_usage": True,
        },
        ov_config=ov_config,
    )

    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print()
    print("[OK] Export finished")
    print(f"  Output: {output_dir}")
    print(f"  Size:   {total_size / 1024**3:.1f} GB")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file() and f.suffix in (".xml", ".bin", ".json"):
            print(f"    {f.relative_to(output_dir)} ({f.stat().st_size / 1024**2:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export Hunyuan Dense V1 / Hy-MT2 to OpenVINO with optimum-intel")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(r"D:\Projects\whisper_cpp_win\Hy-MT2-1.8B"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\Projects\whisper_cpp_win\Hy-MT2-1.8B-ov-optimum"),
    )
    parser.add_argument(
        "--task",
        default="text-generation-with-past",
        choices=["text-generation", "text-generation-with-past"],
    )
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    parser.add_argument("--fp32", action="store_true", help="Export FP32 instead of FP16")
    parser.add_argument("--no-clean", action="store_true", help="Do not delete existing output directory")
    args = parser.parse_args()

    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    convert(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        task=args.task,
        device=args.device,
        fp16=not args.fp32,
        clean=not args.no_clean,
    )


if __name__ == "__main__":
    main()
