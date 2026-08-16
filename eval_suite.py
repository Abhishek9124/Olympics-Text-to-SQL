"""Evaluation & Metrics Suite for Olympics NL-to-SQL.

Measures performance across key Text-to-SQL & LLM metrics:
  1. Execution Accuracy (EX): Checks if generated SQL returns identical result rows as Ground Truth.
  2. Exact Match (EM): Normalized string comparison of generated vs expected SQL.
  3. Syntax Validity Rate: Percentage of queries passing SQLite EXPLAIN verification.
  4. Non-empty Result Rate: Percentage of valid queries returning 1+ rows.
  5. Schema Precision & Recall: Measures column selection overlap against required schema columns.
  6. Latency Benchmarks: Mean, median (p50), and 95th percentile (p95) generation times.

Outputs a formatted report to terminal and saves structured metrics to data/eval_results.json.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Any

# Add project root to Python path
ROOT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from backend import engine, heuristic, ingest, storage, synth

EVAL_DATASET_ID = "eval_bench_suite"


def normalize_sql(sql: str) -> str:
    """Normalize SQL text for exact match comparison."""
    if not sql:
        return ""
    sql = sql.strip().rstrip(";")
    sql = re.sub(r"\s+", " ", sql)
    return sql.upper()


def extract_columns_from_sql(sql: str, available_cols: list[str]) -> set[str]:
    """Extract table columns mentioned in a SQL query."""
    sql_upper = sql.upper()
    found = set()
    for col in available_cols:
        # Match word boundaries around column names
        pattern = r"\b" + re.escape(col.upper()) + r"\b"
        if re.search(pattern, sql_upper):
            found.add(col)
    return found


def compare_result_sets(rows_gen: list[list[Any]], rows_exp: list[list[Any]]) -> bool:
    """Compare two execution result sets regardless of row ordering (unless ordered)."""
    if len(rows_gen) != len(rows_exp):
        return False

    # Convert rows to string tuples for hashing & set comparison
    def row_to_tuple(row: list[Any]) -> tuple:
        return tuple(str(val) if val is not None else "" for val in row)

    tuples_gen = sorted([row_to_tuple(r) for r in rows_gen])
    tuples_exp = sorted([row_to_tuple(r) for r in rows_exp])
    return tuples_gen == tuples_exp


def run_evaluation(num_samples: int = 150, engine_type: str = "auto") -> dict[str, Any]:
    """Execute evaluation benchmark suite."""
    print(f"\n========================================================")
    print(f"      Olympics NL-to-SQL Benchmark & Eval Suite         ")
    print(f"========================================================\n")

    csv_path = ROOT_DIR / "olympics.csv"
    if not csv_path.exists():
        raise FileNotFoundError("olympics.csv not found in project root.")

    db_dir = ROOT_DIR / "data" / "datasets" / EVAL_DATASET_ID
    db_dir.mkdir(parents=True, exist_ok=True)

    print("--> Ingesting dataset and profiling columns...")
    meta = ingest.ingest_csv(EVAL_DATASET_ID, csv_path, "olympics")
    cols = [c["name"] for c in meta["columns"]]
    print(f"    Loaded {meta['row_count']:,} rows | {len(cols)} columns\n")

    print(f"--> Generating {num_samples} test evaluation pairs (NL question -> Ground Truth SQL)...")
    test_pairs = synth.generate_pairs(meta, limit=num_samples)
    print(f"    Generated {len(test_pairs)} evaluation cases.\n")

    print("--> Running Evaluation Suite...")

    total = len(test_pairs)
    valid_count = 0
    exact_matches = 0
    execution_acc_count = 0
    non_empty_count = 0

    latencies_ms: list[float] = []
    schema_precisions: list[float] = []
    schema_recalls: list[float] = []

    eval_details = []

    for idx, (question, expected_sql) in enumerate(test_pairs, 1):
        t0 = time.perf_counter()

        if engine_type == "heuristic":
            gen_sql = heuristic.generate(meta, question)
            eng_used = "heuristic"
        else:
            gen_sql, eng_used = engine.generate(meta, question)

        gen_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(gen_ms)

        # 1. Syntax / Validity Rate
        is_valid, val_reason = engine.validate_sql(EVAL_DATASET_ID, gen_sql)
        if is_valid:
            valid_count += 1

        # 2. Exact Match (EM)
        is_exact = normalize_sql(gen_sql) == normalize_sql(expected_sql)
        if is_exact:
            exact_matches += 1

        # 3. Execution Accuracy (EX) & Non-empty rate
        is_exec_acc = False
        has_rows = False
        gen_cols, gen_rows = [], []
        exp_cols, exp_rows = [], []

        if is_valid:
            try:
                gen_cols, gen_rows = engine.run_sql(EVAL_DATASET_ID, gen_sql)
                exp_cols, exp_rows = engine.run_sql(EVAL_DATASET_ID, expected_sql)

                if gen_rows:
                    has_rows = True
                    non_empty_count += 1

                is_exec_acc = compare_result_sets(gen_rows, exp_rows)
                if is_exec_acc:
                    execution_acc_count += 1

            except Exception:
                is_exec_acc = False

        # 4. Schema Precision & Recall
        expected_cols = extract_columns_from_sql(expected_sql, cols)
        generated_cols = extract_columns_from_sql(gen_sql, cols)

        if generated_cols:
            precision = len(expected_cols.intersection(generated_cols)) / len(generated_cols)
        else:
            precision = 0.0

        if expected_cols:
            recall = len(expected_cols.intersection(generated_cols)) / len(expected_cols)
        else:
            recall = 1.0

        schema_precisions.append(precision)
        schema_recalls.append(recall)

        eval_details.append({
            "id": idx,
            "question": question,
            "expected_sql": expected_sql,
            "generated_sql": gen_sql,
            "engine": eng_used,
            "is_valid": is_valid,
            "is_exact_match": is_exact,
            "is_exec_accuracy": is_exec_acc,
            "gen_rows_count": len(gen_rows),
            "exp_rows_count": len(exp_rows),
            "latency_ms": round(gen_ms, 2),
            "schema_precision": round(precision, 4),
            "schema_recall": round(recall, 4)
        })

    # Aggregate Metrics Calculations
    valid_rate = (valid_count / total) * 100 if total > 0 else 0
    em_rate = (exact_matches / total) * 100 if total > 0 else 0
    ex_rate = (execution_acc_count / total) * 100 if total > 0 else 0
    non_empty_rate = (non_empty_count / total) * 100 if total > 0 else 0

    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0
    sorted_lat = sorted(latencies_ms)
    p50_latency = sorted_lat[int(len(sorted_lat) * 0.50)] if sorted_lat else 0
    p95_latency = sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else 0

    avg_precision = (sum(schema_precisions) / len(schema_precisions)) * 100 if schema_precisions else 0
    avg_recall = (sum(schema_recalls) / len(schema_recalls)) * 100 if schema_recalls else 0

    summary = {
        "dataset": "olympics.csv (120 years, 271k rows)",
        "total_queries_evaluated": total,
        "metrics": {
            "execution_accuracy_ex_pct": round(ex_rate, 2),
            "exact_match_em_pct": round(em_rate, 2),
            "syntax_validity_rate_pct": round(valid_rate, 2),
            "non_empty_result_rate_pct": round(non_empty_rate, 2),
            "schema_precision_pct": round(avg_precision, 2),
            "schema_recall_pct": round(avg_recall, 2),
        },
        "latency_ms": {
            "avg": round(avg_latency, 2),
            "p50": round(p50_latency, 2),
            "p95": round(p95_latency, 2),
        },
        "evaluation_details": eval_details
    }

    # Print Report
    print("\n========================================================")
    print("                EVALUATION REPORT SUMMARY                ")
    print("========================================================")
    print(f" Total Evaluated Queries       : {total}")
    print(f" Execution Accuracy (EX)       : {ex_rate:.2f}% ({execution_acc_count}/{total})")
    print(f" Exact Match Accuracy (EM)     : {em_rate:.2f}% ({exact_matches}/{total})")
    print(f" Syntax Validity Rate          : {valid_rate:.2f}% ({valid_count}/{total})")
    print(f" Non-Empty Result Rate         : {non_empty_rate:.2f}% ({non_empty_count}/{total})")
    print(f" Schema Precision              : {avg_precision:.2f}%")
    print(f" Schema Recall                 : {avg_recall:.2f}%")
    print("--------------------------------------------------------")
    print(f" Generation Latency (Avg)      : {avg_latency:.2f} ms")
    print(f" Generation Latency (p50)      : {p50_latency:.2f} ms")
    print(f" Generation Latency (p95)      : {p95_latency:.2f} ms")
    print("========================================================\n")

    # Save to data/eval_results.json
    out_dir = ROOT_DIR / "data"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"--> Saved evaluation metrics report to {out_file}\n")
    return summary


if __name__ == "__main__":
    samples = 150
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        samples = int(sys.argv[1])
    run_evaluation(num_samples=samples)
