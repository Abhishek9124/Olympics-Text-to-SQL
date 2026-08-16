"""Quick benchmark: evaluate the heuristic engine on synthetic question pairs."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backend import ingest, storage, engine, synth, heuristic

TMP_ID = "bench_eval"
tmp_dir = Path("data/datasets") / TMP_ID
tmp_dir.mkdir(parents=True, exist_ok=True)

print("Ingesting olympics.csv...")
meta = ingest.ingest_csv(TMP_ID, Path("olympics.csv"), "olympics")
print(f"  {meta['row_count']:,} rows, {len(meta['columns'])} columns")

pairs = synth.generate_pairs(meta, limit=600)
print(f"  {len(pairs)} synthetic test pairs\n")

valid = 0
has_rows = 0
latencies = []
errors = []

for question, expected_sql in pairs:
    try:
        t0 = time.perf_counter()
        gen_sql = heuristic.generate(meta, question)
        ok, _ = engine.validate_sql(TMP_ID, gen_sql)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        if ok:
            valid += 1
            cols, rows = engine.run_sql(TMP_ID, gen_sql)
            if rows:
                has_rows += 1
    except Exception as e:
        errors.append(str(e)[:80])

n = len(pairs)
avg_lat = sum(latencies) / len(latencies) if latencies else 0
p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

print("=== Deterministic engine — olympics.csv benchmark ===")
print(f"  Queries evaluated      : {n}")
print(f"  Valid SQL rate         : {valid}/{n}  ({100*valid/n:.1f}%)")
print(f"  Non-empty result rate  : {has_rows}/{n}  ({100*has_rows/n:.1f}%)")
print(f"  Avg generation latency : {avg_lat:.1f} ms")
print(f"  p95 generation latency : {p95_lat:.1f} ms")
if errors:
    print(f"  Errors                 : {len(errors)}")

# Cleanup
import shutil
shutil.rmtree(tmp_dir, ignore_errors=True)
(Path("data/datasets") / f"{TMP_ID}_meta.json").unlink(missing_ok=True)
