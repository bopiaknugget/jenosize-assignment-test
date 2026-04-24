from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

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


def merge_adapter(adapter_dir: Path, base_model_name: str, merged_dir: Path) -> None:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")

    print(f"Merging adapter from: {adapter_dir}")
    print(f"Using base model: {base_model_name}")
    print(f"Saving merged model to: {merged_dir}")

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter_dir),
        torch_dtype=torch_dtype,
        device_map="cpu",
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

    merge_adapter(args.adapter_dir, base_model_name, args.merged_dir)

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
