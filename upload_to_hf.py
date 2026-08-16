"""Hugging Face Hub Exporter & Artifact Publisher.

Allows uploading either:
  1. Dataset: Formatted Olympic Natural Language to SQL evaluation/training dataset
  2. Model: Fine-tuned LoRA Adapter weights (adapter_model.safetensors & adapter_config.json)

Usage:
    # Dry-run validation (no HF token required):
    python upload_to_hf.py dataset --dry_run

    # Push dataset to Hugging Face Hub:
    python upload_to_hf.py dataset --repo_id username/olympics-nl2sql-dataset --token hf_xxx

    # Push LoRA Model Adapter to Hugging Face Hub:
    python upload_to_hf.py model --repo_id username/flan-t5-olympics-sql-adapter --token hf_xxx
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from backend import ingest, synth


def prepare_hf_dataset(num_samples: int = 300) -> tuple[list[dict[str, Any]], str]:
    """Generate and structure dataset into standard Hugging Face JSONL format."""
    csv_path = ROOT_DIR / "olympics.csv"
    if not csv_path.exists():
        raise FileNotFoundError("olympics.csv missing from workspace root.")

    meta = ingest.ingest_csv("hf_export", csv_path, "olympics")
    sig = ingest.schema_signature(meta)
    pairs = synth.generate_pairs(meta, limit=num_samples)

    formatted_rows = []
    for idx, (question, sql) in enumerate(pairs, 1):
        formatted_rows.append({
            "id": f"olympics_{idx:04d}",
            "question": question,
            "sql": sql,
            "db_id": "olympics",
            "schema_signature": sig,
            "dataset_rows": meta["row_count"],
        })

    return formatted_rows, sig


def upload_dataset(
    repo_id: str,
    token: str | None = None,
    num_samples: int = 300,
    dry_run: bool = False
) -> None:
    """Export formatted dataset and push to Hugging Face Hub."""
    print("\n========================================================")
    print("        Hugging Face Hub — Dataset Exporter            ")
    print("========================================================\n")

    rows, sig = prepare_hf_dataset(num_samples)
    print(f"--> Formatted {len(rows)} (NL Question -> SQL) dataset records.")

    out_dir = ROOT_DIR / "data" / "hf_export"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "olympics_nl2sql.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"--> Saved dataset JSONL locally to: {jsonl_path}")

    # Dataset Readme card
    readme_content = f"""---
language:
- en
license: mit
tags:
- text-to-sql
- sqlite
- olympics
- fine-tuning
size_categories:
- n<1K
---

# 🏅 Olympics Natural Language to SQL Dataset

This dataset contains **{len(rows)} synthetic and schema-derived question-SQL pairs** built over 120 years of Olympic history (271,116 rows).

## Dataset Structure

- `question`: Natural language question in plain English.
- `sql`: Ground-truth executable SQLite query.
- `schema_signature`: Summary of table columns, types, and cardinalities.
- `db_id`: Database identifier (`olympics`).

## Schema Signature
```
{sig}
```
"""
    readme_path = out_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    if dry_run:
        print("\n[Dry Run] Validation successful. HF repository files ready at data/hf_export/")
        print("To publish, re-run with: --repo_id <YOUR_HF_USERNAME>/<REPO_NAME> --token <HF_TOKEN>")
        print("========================================================\n")
        return

    if not repo_id:
        raise ValueError("Please provide --repo_id (e.g. username/olympics-nl2sql)")

    try:
        from datasets import Dataset
        print(f"--> Pushing dataset to Hugging Face Hub repo: '{repo_id}'...")
        hf_dataset = Dataset.from_list(rows)
        hf_dataset.push_to_hub(repo_id, token=token)
        print(f"\n🚀 Dataset successfully published! View at: https://huggingface.co/datasets/{repo_id}")
    except Exception as exc:
        print(f"\n--> Failed to publish dataset via datasets library: {exc}")
        print("    You can manually upload the generated file 'data/hf_export/olympics_nl2sql.jsonl' on https://huggingface.co/new-dataset")
    print("========================================================\n")


def upload_model(
    repo_id: str,
    token: str | None = None,
    adapter_dir: Path | None = None,
    dry_run: bool = False
) -> None:
    """Publish LoRA adapter weights to Hugging Face Hub."""
    print("\n========================================================")
    print("        Hugging Face Hub — Model Adapter Publisher      ")
    print("========================================================\n")

    if adapter_dir is None:
        adapter_dir = ROOT_DIR / "data" / "lora_adapters" / "olympics_sql_adapter"

    if not adapter_dir.exists():
        print(f"--> LoRA adapter directory '{adapter_dir}' does not exist.")
        print("    Run 'python lora_finetune.py' first to generate adapter weights.")
        return

    files = list(adapter_dir.glob("*"))
    print(f"--> Found {len(files)} adapter files in {adapter_dir}:")
    for f in files:
        print(f"    - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    if dry_run:
        print("\n[Dry Run] Adapter files validated successfully.")
        print("========================================================\n")
        return

    if not repo_id:
        raise ValueError("Please provide --repo_id (e.g. username/flan-t5-olympics-sql-adapter)")

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        print(f"--> Uploading adapter folder to Hugging Face Hub repo: '{repo_id}'...")
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=str(adapter_dir),
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"\n🚀 Model adapter successfully published! View at: https://huggingface.co/{repo_id}")
    except Exception as exc:
        print(f"\n--> Upload failed: {exc}")
    print("========================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Dataset or Model Adapter to Hugging Face Hub")
    parser.add_argument("target", choices=["dataset", "model"], help="Target artifact to upload")
    parser.add_argument("--repo_id", type=str, default="", help="Hugging Face repo id (username/repository)")
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN", ""), help="Hugging Face User Access Token")
    parser.add_argument("--dry_run", action="store_true", help="Validate format locally without uploading")
    parser.add_argument("--samples", type=int, default=300, help="Number of dataset samples")

    args = parser.parse_args()

    if args.target == "dataset":
        upload_dataset(repo_id=args.repo_id, token=args.token or None, num_samples=args.samples, dry_run=args.dry_run)
    elif args.target == "model":
        upload_model(repo_id=args.repo_id, token=args.token or None, dry_run=args.dry_run)
