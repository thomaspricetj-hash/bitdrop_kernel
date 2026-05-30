import subprocess
import re
import os
import time
import statistics
import sys

ROOT = r"C:\Users\thomas price\Desktop\kernal"
BENCH_SCRIPT = "bench_bitdrop_gpu.py"

TIMEOUT = 45

THROUGHPUT_PATTERNS = [
    r"Throughput:\s+([\d,]+)",
    r"vec/sec[:\s]+([\d,]+)",
    r"([\d,]+)\s+vec/sec",
]

MAX_REASONABLE_THROUGHPUT = 500_000_000

# Internal sampling for stability
INTERNAL_SAMPLES = 3


def _parse_throughput(output: str) -> int:
    for pat in THROUGHPUT_PATTERNS:
        m = re.search(pat, output)
        if m:
            return int(m.group(1).replace(",", ""))
    raise RuntimeError("Could not parse throughput from benchmark output:\n" + output)


def _run_once() -> int:
    """
    Runs the benchmark script exactly once.
    No sleeps, no jitter logic — pure measurement.
    """
    bench_path = os.path.join(ROOT, BENCH_SCRIPT)
    if not os.path.exists(bench_path):
        raise FileNotFoundError(f"Benchmark script not found: {bench_path}")

    try:
        proc = subprocess.run(
            [sys.executable, BENCH_SCRIPT],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Benchmark timed out (kernel hang or deadlock)")

    out = proc.stdout + "\n" + proc.stderr

    if proc.returncode != 0:
        raise RuntimeError(
            f"Benchmark failed with return code {proc.returncode}:\n{out}"
        )

    score = _parse_throughput(out)

    if score <= 0:
        raise RuntimeError(f"Invalid throughput value: {score}")

    if score > MAX_REASONABLE_THROUGHPUT:
        raise RuntimeError(f"Throughput value unrealistic: {score}")

    return score


def measure_throughput() -> int:
    """
    Clean, stable, sweep‑friendly measurement:
    - No sleeps
    - No outlier trimming per call
    - No jitter penalties
    - Just average INTERNAL_SAMPLES runs
    """
    samples = []

    for _ in range(INTERNAL_SAMPLES):
        samples.append(_run_once())

    # Simple average — autotuner handles outliers globally
    return int(statistics.mean(samples))



