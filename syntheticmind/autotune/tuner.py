import time
import traceback
import statistics
import json
import os

from .rewrite import rewrite_kernel
from .build import rebuild_pyd, METRICS_FILE
from .measure import measure_throughput
from .search import generate_configs
from .cache import load_best, save_best


# ============================================================
#  CONFIG
# ============================================================

WARMUP_RUNS = 2
MEASURE_RUNS = 5
OUTLIER_DROP = 1
CRASH_PENALTY = -1e12
MAX_CRASHES = 3
BASE_IMPROVEMENT = 0.25
MIN_IMPROVEMENT = 0.10

SMART_MUTATION_LIMIT = 16
STALL_WINDOW = 12
STALL_MIN_IMPROVEMENT = 0.05


# ============================================================
#  PAYLOAD MULTIPLIER (GPU saturation test)
# ============================================================

PAYLOAD_MULTIPLIER = 1


# ============================================================
#  BATCH SWEEP CONFIG
# ============================================================

BATCH_SWEEP = [1, 2, 4, 8, 16, 32]


# ============================================================
#  HELPERS
# ============================================================

def _load_metrics():
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _register_pressure_helper(metrics):
    if not metrics:
        return None
    regs = metrics["kernels"][0]["registers"]
    if regs > 96:
        return "high"
    if regs > 64:
        return "medium"
    return "low"


def _smem_pressure_helper(metrics):
    if not metrics:
        return None
    smem = metrics["kernels"][0]["shared_mem"]
    if smem > 32 * 1024:
        return "high"
    if smem > 16 * 1024:
        return "medium"
    return "low"


def _occupancy_helper(metrics):
    if not metrics:
        return None
    regs = metrics["kernels"][0]["registers"]
    if regs < 64:
        return "high"
    if regs <= 96:
        return "medium"
    return "low"


def _jitter_helper(samples):
    if len(samples) < 2:
        return "stable"
    mean = statistics.mean(samples)
    stdev = statistics.pstdev(samples)
    if mean > 0 and stdev / mean > 0.20:
        return "unstable"
    return "stable"


def _thermal_drift_helper(history):
    if len(history) < 5:
        return "normal"
    last = [s for s, _ in history[-5:]]
    if last[-1] < last[0] * 0.90:
        return "throttling"
    return "normal"


def _avg(values):
    return sum(values) / len(values) if values else 0.0


def _percent_diff(a, b):
    if b <= 0:
        return 0.0
    return 100.0 * (a - b) / b


def _batch_regime(batch_mult: int) -> str:
    if batch_mult <= 2:
        return "small"
    if batch_mult <= 8:
        return "medium"
    return "large"


# ============================================================
#  SAFE MEASURE
# ============================================================

def measure_with_batch_multiplier(batch_mult: int) -> float:
    total = 0.0
    for _ in range(batch_mult):
        total += measure_throughput()
    return total / batch_mult


# ============================================================
#  INSIGHT ENGINE
# ============================================================

def _build_param_stats(results):
    param_stats = {}
    for cfg, score in results:
        if score <= 0:
            continue
        for k, v in cfg.items():
            if k == "stage":
                continue
            param_stats.setdefault(k, {}).setdefault(v, []).append(score)
    return param_stats


def print_insights(results, best_cfg, best_score):
    print("\n=== PERFORMANCE INSIGHT REPORT ===")

    if not results or best_cfg is None:
        print("Not enough data to generate insights.")
        return

    print(f"Best config: {best_cfg}  ({best_score:,.0f} vec/sec)")

    param_stats = _build_param_stats(results)
    suggestions = []

    for param, val_map in param_stats.items():
        if len(val_map) < 2:
            continue

        avg_by_val = {v: _avg(scores) for v, scores in val_map.items()}
        sorted_vals = sorted(avg_by_val.items(), key=lambda x: x[1], reverse=True)
        best_val, best_val_score = sorted_vals[0]
        worst_val, worst_val_score = sorted_vals[-1]
        diff = _percent_diff(best_val_score, worst_val_score)

        if diff >= 3.0:
            suggestions.append(
                f"- `{param}`: {best_val} beats {worst_val} by ~{diff:.1f}%"
            )

    if not suggestions:
        print("Configs are relatively flat — no strong parameter preferences detected.")
        return

    print("\nKey observations:")
    for s in suggestions:
        print(s)


# ============================================================
#  STALL DETECTION
# ============================================================

def _is_stalled(history, best_score):
    if len(history) < STALL_WINDOW:
        return False

    window = history[-STALL_WINDOW:]
    if any(was_best for _, was_best in window):
        return False

    avg_window = _avg([s for s, _ in window if s > 0])
    if avg_window <= 0 or best_score <= 0:
        return False

    diff = _percent_diff(best_score, avg_window)
    return diff < STALL_MIN_IMPROVEMENT


# ============================================================
#  CONFIG EXECUTION
# ============================================================

def _run_single_config(cfg, seen, results, best_score, best_cfg, history, batch_profiles):
    key = tuple(sorted(cfg.items()))
    if key in seen:
        return None, best_score, best_cfg, False
    seen.add(key)

    print(f"\n=== Testing config ({cfg['stage']}): {cfg} ===")

    rewrite_kernel(cfg)

    if not rebuild_pyd():
        print("Build failed — skipping")
        return None, best_score, best_cfg, False

    scores_by_batch = {}

    for batch_mult in BATCH_SWEEP:
        score = measure_with_batch_multiplier(batch_mult)
        scores_by_batch[batch_mult] = score
        print(f"  Batch x{batch_mult}: {score:,.0f} vec/sec")

        regime = _batch_regime(batch_mult)
        profile = batch_profiles[regime]
        if score > profile["score"]:
            profile["score"] = score
            profile["config"] = dict(cfg)

    score = max(scores_by_batch.values()) if scores_by_batch else CRASH_PENALTY

    if score == CRASH_PENALTY:
        print("Crashed — skipping")
        return None, best_score, best_cfg, False

    metrics = _load_metrics()
    regp = _register_pressure_helper(metrics)
    smemp = _smem_pressure_helper(metrics)
    occ = _occupancy_helper(metrics)
    jitter = _jitter_helper(list(scores_by_batch.values()))
    thermal = _thermal_drift_helper(history)

    print(
        f"Throughput(best batch): {score:,.0f} vec/sec  | "
        f"regs={regp} smem={smemp} occ={occ} jitter={jitter} thermal={thermal}"
    )

    results.append((cfg, score))

    threshold = MIN_IMPROVEMENT if best_score > 1 else BASE_IMPROVEMENT

    was_best = False
    if score > best_score * (1 + threshold / 100):
        best_score = score
        best_cfg = cfg
        was_best = True
        save_best(best_cfg, best_score)
        print(f"🔥 NEW BEST: {best_score:,.0f} vec/sec")

    return score, best_score, best_cfg, was_best


# ============================================================
#  MAIN AUTOTUNE LOOP
# ============================================================

def autotune():
    print("\n=== BitDrop Ultra Autotuner v10 (Batch‑Aware Profiles) ===")

    best = load_best()
    best_score = best["score"] if best else 0
    best_cfg = best["config"] if best else None

    if best_cfg:
        print(f"Loaded best: {best_cfg}  ({best_score:,.0f} vec/sec)")
    else:
        print("No cache found — starting coarse search")

    seen = set()
    results = []
    history = []

    batch_profiles = {
        "small":  {"score": 0.0, "config": None},
        "medium": {"score": 0.0, "config": None},
        "large":  {"score": 0.0, "config": None},
    }

    if best_cfg:
        cached = dict(best_cfg)
        cached["stage"] = "cached"
        score, best_score, best_cfg, was_best = _run_single_config(
            cached, seen, results, best_score, best_cfg, history, batch_profiles
        )
        if score is not None:
            history.append((score, was_best))

    for cfg in generate_configs(best_cfg):
        score, best_score, best_cfg, was_best = _run_single_config(
            cfg, seen, results, best_score, best_cfg, history, batch_profiles
        )
        if score is not None:
            history.append((score, was_best))

    if _is_stalled(history, best_score):
        print("\n=== STALL DETECTED — SMART MUTATION PHASE ===")
    else:
        print("\nNo stall detected — skipping smart phase.")

    print("\n=== AUTOTUNE COMPLETE ===")
    print("Best overall config:", best_cfg)
    print(f"Best overall throughput: {best_score:,.0f} vec/sec")

    print("\n=== BATCH‑AWARE PROFILES ===")
    for regime in ("small", "medium", "large"):
        profile = batch_profiles[regime]
        cfg = profile["config"]
        score = profile["score"]
        if cfg is None:
            print(f"{regime}: no winning config")
        else:
            print(f"{regime}: {cfg}  ({score:,.0f} vec/sec)")

    print_insights(results, best_cfg, best_score)


