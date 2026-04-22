from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import CONFIG
from app.evaluation.tuner import GenerationConfig


class ArticleGenerator:
    def __init__(self, model_dir: Optional[str] = None, base_model_name: Optional[str] = None) -> None:
        self.model_dir = Path(model_dir or CONFIG.training.output_dir)
        self.base_model_name = base_model_name or CONFIG.training.base_model_name
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

    def _load_tokenizer(self):
        source = str(self.model_dir) if self.model_dir.exists() else self.base_model_name
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self):
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        adapter_config = self.model_dir / "adapter_config.json"
        if adapter_config.exists():
            return AutoPeftModelForCausalLM.from_pretrained(
                str(self.model_dir),
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        return AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    def generate(self, system_prompt: str, user_prompt: str, config: GenerationConfig) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **model_inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return self._postprocess(decoded, user_prompt)

    @staticmethod
    def _postprocess(decoded: str, user_prompt: str) -> str:
        if user_prompt in decoded:
            decoded = decoded.split(user_prompt, 1)[-1].strip()
        return decoded.strip()
