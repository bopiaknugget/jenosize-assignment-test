from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

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
    return cfg


def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def format_example(example: dict, tokenizer) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main() -> None:
    args = parse_args()
    cfg = build_training_config(args)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"Using base model: {cfg.base_model_name}")
    print(f"Saving adapter to: {cfg.output_dir}")
    print(f"4-bit loading enabled: {cfg.use_4bit}")

    tokenizer = get_tokenizer(cfg.base_model_name)
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(cfg.train_file),
            "validation": str(cfg.validation_file),
        },
    )

    dataset = dataset.map(
        lambda row: format_example(row, tokenizer),
        remove_columns=dataset["train"].column_names,
    )

    quantization_config = None
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if cfg.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

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

    training_args = TrainingArguments(
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
        bf16=torch.cuda.is_available(),
        fp16=False,
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
