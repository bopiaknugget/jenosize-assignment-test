from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a PEFT adapter into its base model and optionally convert it to GGUF."
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("artifacts/model_adapter"),
        help="Directory containing adapter_config.json and adapter_model.safetensors.",
    )
    parser.add_argument(
        "--base-model-name",
        help="Explicit Hugging Face base model ID. Defaults to adapter_config.json value.",
    )
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=Path("artifacts/model_merged"),
        help="Output directory for the merged Hugging Face model.",
    )
    parser.add_argument(
        "--gguf-path",
        type=Path,
        default=Path("artifacts/model.gguf"),
        help="Target GGUF file path.",
    )
    parser.add_argument(
        "--skip-gguf",
        action="store_true",
        help="Only merge the adapter and save a standard Hugging Face model.",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        type=Path,
        help="Local llama.cpp checkout/build directory. Required for GGUF export unless convert_hf_to_gguf.py is on PATH.",
    )
    parser.add_argument(
        "--outtype",
        default="f16",
        help="GGUF precision passed to convert_hf_to_gguf.py, e.g. f16, bf16, q8_0.",
    )
    parser.add_argument(
        "--quantize",
        help="Optional llama.cpp quantization preset, e.g. Q8_0 or Q4_K_M.",
    )
    parser.add_argument(
        "--quantized-gguf-path",
        type=Path,
        help="Output path for a quantized GGUF. Required when --quantize is used.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help=(
            "Device map for loading the adapter before merge. "
            "Use 'auto' for Colab T4 GPU mapping, 'cuda:0' to force one GPU, or 'cpu' as a fallback."
        ),
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Torch dtype for model loading. 'auto' selects float16 on CUDA and float32 on CPU.",
    )
    parser.add_argument(
        "--max-gpu-memory",
        default="14GiB",
        help="Per-GPU max memory for device_map='auto'. 14GiB leaves headroom on a 16GB Colab T4.",
    )
    parser.add_argument(
        "--max-cpu-memory",
        default="24GiB",
        help="CPU RAM budget used by accelerate when device_map='auto'.",
    )
    return parser.parse_args()


def load_base_model_name(adapter_dir: Path, override: str | None) -> str:
    if override:
        return override

    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Missing adapter config: {adapter_config_path}")

    with adapter_config_path.open("r", encoding="utf-8") as fh:
        adapter_config = json.load(fh)

    base_model_name = adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError("base_model_name_or_path not found in adapter_config.json")
    return base_model_name


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return torch.float16 if torch.cuda.is_available() else torch.float32


def build_max_memory(device_map: str, max_gpu_memory: str, max_cpu_memory: str) -> dict[Any, str] | None:
    if device_map != "auto":
        return None

    max_memory: dict[Any, str] = {"cpu": max_cpu_memory}
    for index in range(torch.cuda.device_count()):
        max_memory[index] = max_gpu_memory
    return max_memory


def merge_adapter(
    adapter_dir: Path,
    base_model_name: str,
    merged_dir: Path,
    device_map: str,
    torch_dtype: torch.dtype,
    max_memory: dict[Any, str] | None,
) -> None:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")

    print(f"Merging adapter from: {adapter_dir}")
    print(f"Using base model: {base_model_name}")
    print(f"Saving merged model to: {merged_dir}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device map: {device_map}")
    print(f"Torch dtype: {torch_dtype}")
    if max_memory:
        print(f"Max memory: {max_memory}")

    model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter_dir),
        torch_dtype=torch_dtype,
        device_map=device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    merged_model = model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))


def resolve_convert_script(llama_cpp_dir: Path | None) -> str:
    if llama_cpp_dir:
        candidates = [
            llama_cpp_dir / "convert_hf_to_gguf.py",
            llama_cpp_dir / "scripts" / "convert_hf_to_gguf.py",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found under {llama_cpp_dir}"
        )

    path_hit = shutil.which("convert_hf_to_gguf.py")
    if path_hit:
        return path_hit

    raise FileNotFoundError(
        "Unable to find convert_hf_to_gguf.py. Pass --llama-cpp-dir pointing to a llama.cpp checkout."
    )


def resolve_quantize_binary(llama_cpp_dir: Path | None) -> str:
    if llama_cpp_dir:
        candidates = [
            llama_cpp_dir / "llama-quantize",
            llama_cpp_dir / "llama-quantize.exe",
            llama_cpp_dir / "build" / "bin" / "Release" / "llama-quantize.exe",
            llama_cpp_dir / "build" / "bin" / "llama-quantize.exe",
            llama_cpp_dir / "build" / "bin" / "llama-quantize",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(f"llama-quantize not found under {llama_cpp_dir}")

    for name in ("llama-quantize", "llama-quantize.exe"):
        path_hit = shutil.which(name)
        if path_hit:
            return path_hit

    raise FileNotFoundError(
        "Unable to find llama-quantize. Pass --llama-cpp-dir pointing to a built llama.cpp tree."
    )


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def convert_to_gguf(
    merged_dir: Path,
    gguf_path: Path,
    outtype: str,
    llama_cpp_dir: Path | None,
) -> None:
    convert_script = resolve_convert_script(llama_cpp_dir)
    gguf_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        convert_script,
        str(merged_dir),
        "--outfile",
        str(gguf_path),
        "--outtype",
        outtype,
    ]
    run_command(command)


def quantize_gguf(
    input_path: Path,
    output_path: Path,
    quantize: str,
    llama_cpp_dir: Path | None,
) -> None:
    quantize_binary = resolve_quantize_binary(llama_cpp_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [quantize_binary, str(input_path), str(output_path), quantize]
    run_command(command)


def main() -> None:
    args = parse_args()
    base_model_name = load_base_model_name(args.adapter_dir, args.base_model_name)
    torch_dtype = resolve_torch_dtype(args.torch_dtype)
    max_memory = build_max_memory(args.device_map, args.max_gpu_memory, args.max_cpu_memory)

    merge_adapter(
        args.adapter_dir,
        base_model_name,
        args.merged_dir,
        args.device_map,
        torch_dtype,
        max_memory,
    )

    if args.skip_gguf:
        print("Skipping GGUF conversion as requested.")
        return

    convert_to_gguf(args.merged_dir, args.gguf_path, args.outtype, args.llama_cpp_dir)

    if args.quantize:
        if not args.quantized_gguf_path:
            raise ValueError("--quantized-gguf-path is required when --quantize is used")
        quantize_gguf(
            args.gguf_path,
            args.quantized_gguf_path,
            args.quantize,
            args.llama_cpp_dir,
        )


if __name__ == "__main__":
    main()
