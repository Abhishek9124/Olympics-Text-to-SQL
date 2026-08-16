"""LoRA / QLoRA Fine-Tuning Script for Text-to-SQL LLMs.

Demonstrates Parameter-Efficient Fine-Tuning (PEFT) with LoRA / 4-bit QLoRA:
  - Base Models supported: Llama 3.2-3B, Qwen2.5-coder, google/flan-t5-base/small
  - LoRA Adapter: r=16, lora_alpha=32, lora_dropout=0.05
  - Quantization: 4-bit NF4 (bitsandbytes) for VRAM efficiency (QLoRA)
  - Saves adapter weights (adapter_model.safetensors & adapter_config.json)

Usage:
    python lora_finetune.py --base_model google/flan-t5-small --epochs 5
    python lora_finetune.py --base_model meta-llama/Llama-3.2-3B-Instruct --qlora
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from backend import ingest, synth

DEFAULT_BASE_MODEL = os.getenv("LORA_BASE_MODEL", "google/flan-t5-small")
OUTPUT_ADAPTER_DIR = ROOT_DIR / "data" / "lora_adapters" / "olympics_sql_adapter"


def setup_lora_model(
    base_model_name: str,
    use_qlora: bool = False,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05
) -> tuple[Any, Any]:
    """Load base model (optional 4-bit QLoRA quantization) and attach LoRA adapter."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError:
        raise ImportError(
            "peft package is required for LoRA fine-tuning. Run: pip install peft"
        )

    print(f"--> Loading tokenizer for {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detect model type (Seq2Seq vs CausalLM)
    is_seq2seq = any(k in base_model_name.lower() for k in ["t5", "bart", "pegasus"])
    task_type = TaskType.SEQ_2_SEQ_LM if is_seq2seq else TaskType.CAUSAL_LM

    # Quantization Config (4-bit QLoRA via bitsandbytes)
    quantization_config = None
    if use_qlora:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )
            print("--> Enabled 4-bit NF4 QLoRA quantization (bitsandbytes).")
        except Exception as e:
            print(f"--> Warning: BitsAndBytes 4-bit load failed ({e}), falling back to standard precision.")

    print(f"--> Loading base model '{base_model_name}'...")
    device_map = "auto" if (torch.cuda.is_available() and use_qlora) else None
    
    if is_seq2seq:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map=device_map,
        )
        target_modules = ["q", "v", "k", "o"]
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map=device_map,
        )
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Define LoRA Configuration
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=task_type,
    )

    print(f"--> Wrapping model with PEFT LoRA (r={r}, alpha={alpha})...")
    model = get_peft_model(base_model, lora_config)
    
    # Print Trainable Parameter stats
    trainable_params, all_param = model.get_nb_trainable_parameters()
    pct = (trainable_params / all_param) * 100
    print(f"    Trainable params: {trainable_params:,} / {all_param:,} ({pct:.2f}%)")

    return model, tokenizer


def train_lora(
    base_model_name: str = DEFAULT_BASE_MODEL,
    use_qlora: bool = False,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 3e-4,
    num_samples: int = 300,
) -> None:
    """Execute LoRA / QLoRA supervised fine-tuning loop."""
    print("\n========================================================")
    print("      Olympics Text-to-SQL LoRA / QLoRA Trainer        ")
    print("========================================================\n")

    csv_path = ROOT_DIR / "olympics.csv"
    if not csv_path.exists():
        raise FileNotFoundError("olympics.csv not found in project root.")

    print("--> Preparing training dataset...")
    meta = ingest.ingest_csv("lora_train", csv_path, "olympics")
    sig = ingest.schema_signature(meta)
    pairs = synth.generate_pairs(meta, limit=num_samples)
    print(f"    Generated {len(pairs)} synthetic (Question -> SQL) training pairs.")

    model, tokenizer = setup_lora_model(base_model_name, use_qlora=use_qlora)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not use_qlora:
        model.to(device)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"\n--> Starting LoRA fine-tuning on {device} for {epochs} epochs...")
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            inputs_text = [
                f"translate question to SQL | schema: {sig} | question: {q}"
                for q, _ in batch_pairs
            ]
            targets_text = [s for _, s in batch_pairs]

            inputs = tokenizer(
                inputs_text,
                padding=True,
                truncation=True,
                max_length=192,
                return_tensors="pt"
            ).to(device)

            targets = tokenizer(
                text_target=targets_text,
                padding=True,
                truncation=True,
                max_length=160,
                return_tensors="pt"
            ).to(device)

            labels = targets["input_ids"]
            labels[labels == tokenizer.pad_token_id] = -100

            optimizer.zero_grad()
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"    Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    elapsed = time.perf_counter() - t_start
    print(f"\n--> Fine-tuning completed in {elapsed:.1f} seconds.")

    # Save LoRA Adapter
    OUTPUT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"--> Saving LoRA PEFT adapter weights to {OUTPUT_ADAPTER_DIR}...")
    model.save_pretrained(OUTPUT_ADAPTER_DIR)
    tokenizer.save_pretrained(OUTPUT_ADAPTER_DIR)
    print("    Save successful! Files created:")
    for p in OUTPUT_ADAPTER_DIR.iterdir():
        print(f"    - {p.name}")
    print("========================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA/QLoRA Fine-Tuning for Olympics NL-to-SQL")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--qlora", action="store_true", help="Enable 4-bit NF4 QLoRA")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--samples", type=int, default=300)

    args = parser.parse_args()
    train_lora(
        base_model_name=args.base_model,
        use_qlora=args.qlora,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_samples=args.samples,
    )
