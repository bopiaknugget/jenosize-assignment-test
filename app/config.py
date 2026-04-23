from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class ModelCatalog:
    accuracy: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    speed: str = "Qwen/Qwen2.5-3B-Instruct"
    balance: str = "Qwen/Qwen2.5-7B-Instruct"
    embedding: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class TrainingConfig:
    base_model_name: str = "Qwen/Qwen2.5-3B-Instruct" #using speed strategy optimized for running on T4
    train_file: Path = Path("data/training/train.jsonl")
    validation_file: Path = Path("data/training/val.jsonl")
    output_dir: Path = Path("artifacts/model_adapter")
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 2
    learning_rate: float = 2e-4
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_4bit: bool = True


@dataclass
class RetrievalConfig:
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 120
    default_top_k: int = 4


@dataclass
class GenerationRuntimeConfig:
    article_length: str = "900-1200 words"
    max_new_tokens: int = 900
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    single_retry_score_threshold: float = 0.78


@dataclass
class AppConfig:
    models: ModelCatalog = field(default_factory=ModelCatalog)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    runtime: GenerationRuntimeConfig = field(default_factory=GenerationRuntimeConfig)

    def model_options(self) -> Dict[str, str]:
        return {
            "accuracy": self.models.accuracy,
            "speed": self.models.speed,
            "balance": self.models.balance,
        }

    def resolve_generation_model(self, strategy_or_model_name: str | None = None) -> str:
        if not strategy_or_model_name:
            return self.training.base_model_name

        candidate = strategy_or_model_name.strip()
        return self.model_options().get(candidate.lower(), candidate)

    def resolve_embedding_model(self, strategy_or_model_name: str | None = None) -> str:
        if not strategy_or_model_name:
            return self.retrieval.embedding_model_name

        candidate = strategy_or_model_name.strip()
        if candidate.lower() in {"default", "embedding"}:
            return self.models.embedding
        return candidate

    @classmethod
    def from_env(cls) -> "AppConfig":
        config = cls()

        base_choice = os.getenv("BASE_MODEL_NAME") or os.getenv("MODEL_STRATEGY")
        config.training.base_model_name = config.resolve_generation_model(base_choice)

        embedding_choice = os.getenv("EMBEDDING_MODEL_NAME") or os.getenv("EMBEDDING_MODEL_STRATEGY")
        config.retrieval.embedding_model_name = config.resolve_embedding_model(embedding_choice)

        adapter_dir = (
            os.getenv("MODEL_ADAPTER_DIR")
            or os.getenv("FINETUNED_MODEL_DIR")
            or os.getenv("ADAPTER_DIR")
        )
        if adapter_dir:
            config.training.output_dir = Path(adapter_dir)

        use_4bit = os.getenv("USE_4BIT")
        if use_4bit is not None:
            config.training.use_4bit = use_4bit.strip().lower() in {"1", "true", "yes", "y", "on"}

        return config


CONFIG = AppConfig.from_env()
