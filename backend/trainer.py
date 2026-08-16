"""Per-dataset fine-tuning of a small seq2seq model (CPU-friendly).

When a CSV is uploaded the user can "train" a model on it. We:

  1. Generate schema-derived synthetic (question, SQL) pairs (synth.py).
  2. Fine-tune ``Salesforce/codet5-small`` on them with a plain PyTorch loop.
  3. Stream progress (epoch/step/loss) to status.json so the UI can show it.
  4. Save the trained model under the dataset's ``model/`` dir.

Everything degrades gracefully: if torch/transformers aren't importable the job
records state ``unavailable`` and serving falls back to the deterministic
generator. Training runs in a background thread started by the API.
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Optional

from . import storage, synth

BASE_MODEL = os.getenv("AISQL_BASE_MODEL", "google/flan-t5-small")
EPOCHS = int(os.getenv("AISQL_TRAIN_EPOCHS", "12"))
MAX_PAIRS = int(os.getenv("AISQL_TRAIN_MAX_PAIRS", "600"))
# Batch is auto-scaled up on GPU (see _train_worker); this is the CPU default.
BATCH = int(os.getenv("AISQL_TRAIN_BATCH", "16"))
GPU_BATCH = int(os.getenv("AISQL_TRAIN_GPU_BATCH", "32"))
MAX_IN = 192
MAX_OUT = 160
LR = float(os.getenv("AISQL_TRAIN_LR", "3e-4"))

# Early stopping: stop once the loss is genuinely tiny or has truly plateaued,
# so we don't burn the remaining epochs once the model has converged. Kept
# conservative (low loss target + a few epochs of patience) so we never trade
# model quality for speed — the real wall-clock wins come from mixed precision,
# dynamic padding and the larger GPU batch below.
EARLY_STOP_LOSS = float(os.getenv("AISQL_EARLY_STOP_LOSS", "0.015"))
EARLY_STOP_PATIENCE = int(os.getenv("AISQL_EARLY_STOP_PATIENCE", "3"))
MIN_EPOCHS = int(os.getenv("AISQL_TRAIN_MIN_EPOCHS", "6"))

# dataset_id -> bool, guards against concurrent duplicate training.
_RUNNING: dict[str, bool] = {}
_LOCK = threading.Lock()


def build_input(schema_sig: str, question: str) -> str:
    return f"translate question to SQL | schema: {schema_sig} | question: {question}"


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def is_trained(dataset_id: str) -> bool:
    return (storage.model_dir(dataset_id) / "config.json").exists()


def start_training(dataset_id: str) -> dict[str, Any]:
    """Kick off training in a background thread (no-op if already running)."""
    with _LOCK:
        if _RUNNING.get(dataset_id):
            return storage.load_status(dataset_id)
        _RUNNING[dataset_id] = True

    status = {
        "state": "pending",
        "mode": "fine-tune",
        "message": "Queued for training…",
        "progress": 0.0,
        "epoch": 0,
        "total_epochs": EPOCHS,
        "step": 0,
        "total_steps": 0,
        "loss": None,
        "base_model": BASE_MODEL,
        "started_at": _now(),
        "finished_at": None,
        "engine": None,
    }
    storage.save_status(dataset_id, status)

    t = threading.Thread(target=_train_worker, args=(dataset_id,), daemon=True)
    t.start()
    return status


def _train_worker(dataset_id: str) -> None:
    status = storage.load_status(dataset_id)

    def update(**kw: Any) -> None:
        status.update(kw)
        storage.save_status(dataset_id, status)

    try:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            update(
                state="unavailable",
                message=f"ML stack unavailable ({exc}). Using deterministic SQL engine.",
                engine="deterministic",
                finished_at=_now(),
            )
            return

        meta = storage.load_meta(dataset_id)
        if not meta:
            update(state="failed", message="Dataset metadata missing.", finished_at=_now())
            return

        from .ingest import schema_signature

        sig = schema_signature(meta)
        pairs = synth.generate_pairs(meta, limit=MAX_PAIRS)
        if not pairs:
            update(state="failed", message="Could not derive training examples.", finished_at=_now())
            return

        torch.set_num_threads(max(1, os.cpu_count() or 1))
        on_gpu = torch.cuda.is_available()
        device = torch.device("cuda" if on_gpu else "cpu")
        dev_name = torch.cuda.get_device_name(0) if on_gpu else "CPU"
        batch = GPU_BATCH if on_gpu else BATCH

        # GPU fast paths: autotune kernels + allow TF32 matmuls (big speedup on
        # Ampere cards like the RTX 3050, no meaningful accuracy loss here).
        if on_gpu:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Mixed precision: T5/Flan-T5 overflows in fp16 (it was trained in
        # bf16), which silently corrupts the fine-tune. Prefer bf16 — supported
        # on Ampere (RTX 3050) and numerically safe, no loss scaling needed.
        # Only fall back to fp16+GradScaler if bf16 is unavailable.
        use_bf16 = on_gpu and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        precision = "bf16" if use_bf16 else ("fp16" if on_gpu else "fp32")

        update(state="running", message=f"Loading base model on {dev_name}…",
               pairs=len(pairs), device=device.type)

        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)
        model.train()

        # Pre-tokenize once WITHOUT padding, then sort by input length so each
        # batch groups similar-length sequences. With per-batch (dynamic)
        # padding this avoids padding every row to the global max — far fewer
        # wasted tokens, so each step is faster.
        inputs = [build_input(sig, q) for q, _ in pairs]
        targets = [s for _, s in pairs]
        enc_ids = tok(inputs, max_length=MAX_IN, truncation=True)["input_ids"]
        lab_ids = tok(text_target=targets, max_length=MAX_OUT, truncation=True)["input_ids"]

        order = sorted(range(len(enc_ids)), key=lambda i: len(enc_ids[i]))
        enc_ids = [enc_ids[i] for i in order]
        lab_ids = [lab_ids[i] for i in order]
        pad_id = tok.pad_token_id

        def make_batch(rows_in, rows_out):
            in_max = max(len(r) for r in rows_in)
            out_max = max(len(r) for r in rows_out)
            ids, attn, labs = [], [], []
            for ri, ro in zip(rows_in, rows_out):
                ids.append(ri + [pad_id] * (in_max - len(ri)))
                attn.append([1] * len(ri) + [0] * (in_max - len(ri)))
                labs.append(ro + [-100] * (out_max - len(ro)))
            return (
                torch.tensor(ids, device=device),
                torch.tensor(attn, device=device),
                torch.tensor(labs, device=device),
            )

        optim = torch.optim.AdamW(model.parameters(), lr=LR)
        # GradScaler is only needed/valid for fp16; bf16 has fp32's exponent
        # range so it trains without scaling.
        scaler = torch.cuda.amp.GradScaler(enabled=(on_gpu and not use_bf16))
        n = len(enc_ids)
        # Batch start offsets, shuffled each epoch (keeps length-grouping intact).
        starts = list(range(0, n, batch))
        steps_per_epoch = len(starts)
        total_steps = steps_per_epoch * EPOCHS
        update(total_steps=total_steps, batch=batch, precision=precision,
               message=f"Training on {dev_name} ({precision})…")

        import random as _random
        rng = _random.Random(0)
        step = 0
        best_loss = float("inf")
        stale = 0
        stopped_early = False
        for epoch in range(EPOCHS):
            rng.shuffle(starts)
            epoch_loss = 0.0
            for s in starts:
                rin = enc_ids[s : s + batch]
                rout = lab_ids[s : s + batch]
                input_ids, attn, labels = make_batch(rin, rout)
                optim.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=on_gpu):
                    out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                if scaler.is_enabled():
                    scaler.scale(out.loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    out.loss.backward()
                    optim.step()
                step += 1
                epoch_loss += float(out.loss.item())
                if step % 2 == 0 or step == total_steps:
                    update(
                        epoch=epoch + 1,
                        step=step,
                        loss=round(float(out.loss.item()), 4),
                        progress=round(step / total_steps, 3),
                    )

            # --- early stopping on mean epoch loss ---
            mean_loss = epoch_loss / max(1, steps_per_epoch)
            if mean_loss < best_loss - 1e-3:
                best_loss = mean_loss
                stale = 0
            else:
                stale += 1
            # Never stop before MIN_EPOCHS — guards against bailing out while
            # the loss is still dropping fast on these easy templated examples.
            converged = mean_loss <= EARLY_STOP_LOSS or stale >= EARLY_STOP_PATIENCE
            if epoch + 1 >= MIN_EPOCHS and converged:
                stopped_early = True
                update(
                    epoch=epoch + 1,
                    progress=round(step / total_steps, 3),
                    message=f"Converged early at epoch {epoch + 1} (loss {mean_loss:.4f}).",
                )
                break

        update(message="Saving model…", progress=0.99, stopped_early=stopped_early)
        model.eval()
        mdir = storage.model_dir(dataset_id)
        mdir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(mdir))
        tok.save_pretrained(str(mdir))

        update(
            state="done",
            message=f"Fine-tuned {BASE_MODEL} on {len(pairs)} examples.",
            engine=f"fine-tuned · {BASE_MODEL}",
            progress=1.0,
            finished_at=_now(),
        )
    except Exception as exc:  # noqa: BLE001
        update(
            state="failed",
            message=f"Training failed: {exc}. Falling back to deterministic engine.",
            engine="deterministic",
            finished_at=_now(),
            traceback=traceback.format_exc()[-1500:],
        )
    finally:
        with _LOCK:
            _RUNNING.pop(dataset_id, None)


# --------------------------------------------------------------------------- #
# Serving: lazy-load a trained model per dataset and generate SQL (100% Local).
# --------------------------------------------------------------------------- #
_MODELS: dict[str, Any] = {}
_TOKS: dict[str, Any] = {}


def generate_sql(dataset_id: str, schema_sig: str, question: str) -> Optional[str]:
    """Return SQL from the fine-tuned model, or None if it isn't available."""
    if not is_trained(dataset_id):
        return None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
    except Exception:  # noqa: BLE001
        return None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dataset_id not in _MODELS:
            mdir = str(storage.model_dir(dataset_id))
            tok = AutoTokenizer.from_pretrained(mdir, local_files_only=False)
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(mdir).to(device).eval()
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(mdir).to(device).eval()
            _TOKS[dataset_id] = tok
            _MODELS[dataset_id] = model

        tok, model = _TOKS[dataset_id], _MODELS[dataset_id]
        text = build_input(schema_sig, question)
        ids = tok(text, max_length=MAX_IN, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            if hasattr(model, "config") and getattr(model.config, "is_encoder_decoder", False):
                out = model.generate(**ids, max_length=MAX_OUT, num_beams=4, early_stopping=True)
                sql = tok.decode(out[0], skip_special_tokens=True).strip()
            else:
                out = model.generate(**ids, max_new_tokens=128, do_sample=False)
                gen_text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
                sql = gen_text.strip()

        return sql or None
    except Exception:  # noqa: BLE001
        return None

