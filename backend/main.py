"""AI SQL Assistant — FastAPI backend.

Upload any CSV -> it's profiled and loaded into SQLite -> optionally fine-tune a
small model on it -> ask questions in plain English and get live SQL + results.
"""
from __future__ import annotations

import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import engine, ingest, storage, synth, trainer

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_CSV = BASE_DIR / "olympics.csv"  # bundled "try a sample" dataset, if present

# Upload cap — keeps ingestion + (CPU/4 GB-GPU) training fast. Configurable.
MAX_UPLOAD_MB = int(os.getenv("AISQL_MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

app = FastAPI(title="AI SQL Assistant API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class QueryRequest(BaseModel):
    question: str
    dataset_id: Optional[str] = None


class QueryResponse(BaseModel):
    question: str
    dataset_id: str
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    elapsed_ms: int
    engine: str
    error: Optional[str] = None


class SampleRequest(BaseModel):
    train: bool = False


class ExecuteRequest(BaseModel):
    sql: str
    dataset_id: Optional[str] = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resolve(dataset_id: Optional[str]) -> str:
    ds = dataset_id or storage.get_active_id()
    if not ds or not storage.load_meta(ds):
        raise HTTPException(status_code=404, detail="No active dataset. Upload a CSV first.")
    return ds


def _ml_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _compute_stats(meta: dict[str, Any]) -> dict[str, Any]:
    db = storage.db_path(meta["id"])
    table = meta["table"]
    cols = meta["columns"]
    measures = [c for c in cols if c["role"] == "measure"]
    categories = [c for c in cols if c["role"] == "category"]

    def scalar(sql: str, default: Any = None) -> Any:
        try:
            with closing(sqlite3.connect(str(db))) as conn:
                row = conn.execute(sql).fetchone()
            return row[0] if row and row[0] is not None else default
        except Exception:  # noqa: BLE001
            return default

    headline = [
        {"label": "Rows", "value": meta["row_count"]},
        {"label": "Columns", "value": len(cols)},
    ]
    if categories:
        c = categories[0]
        headline.append({
            "label": f"Distinct {c['label']}",
            "value": scalar(f'SELECT COUNT(DISTINCT "{c["name"]}") FROM "{table}"', 0),
        })
    for m in measures[:2]:
        avg = scalar(f'SELECT ROUND(AVG("{m["name"]}"),2) FROM "{table}"')
        if avg is not None:
            headline.append({"label": f"Avg {m['label']}", "value": avg})
    headline.append({"label": "Numeric cols", "value": len(measures)})

    top_dim = None
    if categories:
        c = categories[0]
        items = []
        try:
            with closing(sqlite3.connect(str(db))) as conn:
                for key, cnt in conn.execute(
                    f'SELECT "{c["name"]}", COUNT(*) AS n FROM "{table}" '
                    f'GROUP BY "{c["name"]}" ORDER BY n DESC LIMIT 6'
                ):
                    items.append({"key": str(key), "count": cnt})
        except Exception:  # noqa: BLE001
            pass
        top_dim = {"label": c["label"], "items": items}

    return {
        "row_count": meta["row_count"],
        "column_count": len(cols),
        "n_measures": len(measures),
        "n_categories": len(categories),
        "headline": headline[:6],
        "top_dim": top_dim,
    }


def _meta_public(meta: dict[str, Any]) -> dict[str, Any]:
    """Schema view for the UI (drops nothing sensitive, just curates)."""
    return {
        "id": meta["id"],
        "name": meta["name"],
        "table": meta["table"],
        "row_count": meta["row_count"],
        "columns": [
            {
                "name": c["name"],
                "label": c["label"],
                "type": c["type"],
                "role": c["role"],
                "distinct": c.get("distinct"),
            }
            for c in meta["columns"]
        ],
        "sample_rows": meta.get("sample_rows", []),
        "created_at": meta.get("created_at"),
    }


def _ingest_and_register(source: Path, name: str) -> dict[str, Any]:
    dataset_id = storage.new_dataset_id()
    storage.dataset_dir(dataset_id).mkdir(parents=True, exist_ok=True)
    meta = ingest.ingest_csv(dataset_id, source, name)
    storage.set_active(dataset_id)
    storage.save_status(dataset_id, {"state": "none", "engine": "deterministic"})
    return meta


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict:
    active = storage.get_active_id()
    return {
        "status": "ok",
        "ml_available": _ml_available(),
        "local_only": True,  # no external APIs — everything runs on this machine
        "active_dataset": active,
        "datasets": len(storage.list_datasets()),
        "sample_available": SAMPLE_CSV.exists(),
        "max_upload_mb": MAX_UPLOAD_MB,
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(data) / 1024 / 1024:.1f} MB). "
            f"Maximum upload size is {MAX_UPLOAD_MB} MB.",
        )
    dataset_id = storage.new_dataset_id()
    storage.dataset_dir(dataset_id).mkdir(parents=True, exist_ok=True)
    dest = storage.csv_path(dataset_id)
    dest.write_bytes(data)
    try:
        meta = ingest.ingest_csv(dataset_id, dest, Path(file.filename).stem)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")
    storage.set_active(dataset_id)
    storage.save_status(dataset_id, {"state": "none", "engine": "deterministic"})
    return {
        "dataset_id": dataset_id,
        "schema": _meta_public(meta),
        "stats": _compute_stats(meta),
        "examples": synth.generate_examples(meta),
        "ml_available": _ml_available(),
    }


@app.post("/api/sample")
def load_sample(req: SampleRequest) -> dict:
    if not SAMPLE_CSV.exists():
        raise HTTPException(status_code=404, detail="No bundled sample dataset available.")
    meta = _ingest_and_register(SAMPLE_CSV, "Olympics (sample)")
    if req.train:
        trainer.start_training(meta["id"])
    return {
        "dataset_id": meta["id"],
        "schema": _meta_public(meta),
        "stats": _compute_stats(meta),
        "examples": synth.generate_examples(meta),
        "ml_available": _ml_available(),
    }


@app.post("/api/datasets/{dataset_id}/train")
def train(dataset_id: str) -> dict:
    if not storage.load_meta(dataset_id):
        raise HTTPException(status_code=404, detail="Unknown dataset.")
    if not _ml_available():
        status = {
            "state": "unavailable",
            "message": "PyTorch/transformers not installed — using the deterministic engine.",
            "engine": "deterministic",
        }
        storage.save_status(dataset_id, status)
        return status
    return trainer.start_training(dataset_id)


@app.get("/api/datasets/{dataset_id}/status")
def train_status(dataset_id: str) -> dict:
    if not storage.load_meta(dataset_id):
        raise HTTPException(status_code=404, detail="Unknown dataset.")
    return storage.load_status(dataset_id)


@app.get("/api/datasets")
def datasets() -> dict:
    return {"datasets": storage.list_datasets(), "active": storage.get_active_id()}


@app.post("/api/datasets/{dataset_id}/activate")
def activate(dataset_id: str) -> dict:
    if not storage.load_meta(dataset_id):
        raise HTTPException(status_code=404, detail="Unknown dataset.")
    storage.set_active(dataset_id)
    return {"active": dataset_id}


@app.get("/api/schema")
def schema(dataset_id: Optional[str] = None) -> dict:
    ds = _resolve(dataset_id)
    return _meta_public(storage.load_meta(ds))


@app.get("/api/stats")
def stats(dataset_id: Optional[str] = None) -> dict:
    ds = _resolve(dataset_id)
    return _compute_stats(storage.load_meta(ds))


@app.get("/api/examples")
def examples(dataset_id: Optional[str] = None) -> dict:
    ds = _resolve(dataset_id)
    return {"examples": synth.generate_examples(storage.load_meta(ds))}


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    ds = _resolve(req.dataset_id)
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    meta = storage.load_meta(ds)

    start = time.perf_counter()
    sql, eng = engine.generate(meta, question)
    columns: list[str] = []
    rows: list[list[Any]] = []
    error: Optional[str] = None
    try:
        columns, rows = engine.run_sql(ds, sql)
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return QueryResponse(
        question=question,
        dataset_id=ds,
        sql=sql,
        columns=columns,
        rows=rows,
        row_count=len(rows),
        elapsed_ms=elapsed_ms,
        engine=eng,
        error=error,
    )


@app.post("/api/execute", response_model=QueryResponse)
def execute(req: ExecuteRequest) -> QueryResponse:
    """Run user-edited SQL directly (still validated as a read-only SELECT)."""
    ds = _resolve(req.dataset_id)
    sql = (req.sql or "").strip()
    if not sql:
        raise HTTPException(status_code=400, detail="SQL must not be empty.")

    start = time.perf_counter()
    columns: list[str] = []
    rows: list[list[Any]] = []
    error: Optional[str] = None
    try:
        columns, rows = engine.run_sql(ds, sql)
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return QueryResponse(
        question="(manual SQL)",
        dataset_id=ds,
        sql=sql,
        columns=columns,
        rows=rows,
        row_count=len(rows),
        elapsed_ms=elapsed_ms,
        engine="manual",
        error=error,
    )


# Serve frontend static assets if built (production / Docker deployment)
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
if FRONTEND_DIST.exists():
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="static")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        file_path = FRONTEND_DIST / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIST / "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)

