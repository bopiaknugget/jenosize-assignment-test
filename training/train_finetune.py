from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from app.config import CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face article-generation model with LoRA.")
    parser.add_argument(
        "--model-strategy",
        choices=["accuracy", "speed", "balance"],
        help="Shortcut strategy that resolves to a predefined Hugging Face model ID.",
    )
    parser.add_argument(
        "--base-model-name",
        help="Exact Hugging Face model ID. Overrides --model-strategy when both are provided.",
    )
    parser.add_argument(
        "--output-dir",
        help="Adapter output directory. Defaults to app/config.py or MODEL_ADAPTER_DIR env.",
    )
    parser.add_argument(
        "--use-4bit",
        choices=["true", "false"],
        help="Override 4-bit QLoRA loading. Example: --use-4bit false",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        help="Maximum sequence length for SFT tokenization. Defaults to app/config.py.",
    )
    parser.add_argument(
        "--dataset-num-proc",
        type=int,
        default=1,
        help="Number of processes for dataset formatting. Keep 1 on Windows if multiprocessing is unstable.",
    )
    return parser.parse_args()


def build_training_config(args: argparse.Namespace):
    cfg = copy.deepcopy(CONFIG.training)
    model_choice = args.base_model_name or args.model_strategy
    if model_choice:
        cfg.base_model_name = CONFIG.resolve_generation_model(model_choice)
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    if args.use_4bit:
        cfg.use_4bit = args.use_4bit == "true"
    if args.max_seq_length is not None:
        if args.max_seq_length <= 0:
            raise ValueError("--max-seq-length must be greater than 0.")
        cfg.max_seq_length = args.max_seq_length
    return cfg


def validate_training_files(cfg) -> None:
    missing = [path for path in (cfg.train_file, cfg.validation_file) if not path.exists()]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing training data: {paths}. Run training/prepare_dataset.py first.")


def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def format_batch(batch: dict[str, list[Any]], tokenizer) -> dict[str, list[str]]:
    return {
        "text": [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages in batch["messages"]
        ]
    }


def get_compute_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def main() -> None:
    args = parse_args()
    cfg = build_training_config(args)
    os.makedirs(cfg.output_dir, exist_ok=True)
    validate_training_files(cfg)

    print(f"Using base model: {cfg.base_model_name}")
    print(f"Saving adapter to: {cfg.output_dir}")
    print(f"4-bit loading enabled: {cfg.use_4bit}")
    print(f"Max sequence length: {cfg.max_seq_length}")

    tokenizer = get_tokenizer(cfg.base_model_name)
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(cfg.train_file),
            "validation": str(cfg.validation_file),
        },
    )

    dataset = dataset.map(
        format_batch,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        num_proc=max(1, args.dataset_num_proc),
        remove_columns=dataset["train"].column_names,
        desc="Applying chat templates",
    )

    quantization_config = None
    torch_dtype = get_compute_dtype()
    if cfg.use_4bit:
        if not torch.cuda.is_available():
            print("4-bit loading requires CUDA; falling back to full-precision CPU loading.")
            cfg.use_4bit = False
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        bf16=torch_dtype == torch.bfloat16,
        fp16=torch_dtype == torch.float16,
        optim="paged_adamw_8bit" if cfg.use_4bit else "adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        dataset_num_proc=max(1, args.dataset_num_proc),
        dataset_text_field="text",
        max_length=cfg.max_seq_length,
        packing=False,
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config
    )

    trainer.train()
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))
    print(f"Training complete. Saved artifacts to {cfg.output_dir}")


if __name__ == "__main__":
    main()
