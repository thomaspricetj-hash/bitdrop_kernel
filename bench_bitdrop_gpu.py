import sys
import time
import os
import csv
import numpy as np
import argparse

# -------------------------------------------------------------------
# Load correct bitdrop_gpu.pyd
# -------------------------------------------------------------------
PYD_PATH = r"C:\Users\thomas price\Desktop\kernal\build\python\Release"
if PYD_PATH not in sys.path:
    sys.path.insert(0, PYD_PATH)

import bitdrop_gpu


# -------------------------------------------------------------------
# Env / arg helpers
# -------------------------------------------------------------------
def get_env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def get_env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def get_env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


# -------------------------------------------------------------------
# Bank builders
# -------------------------------------------------------------------
def build_threshold_bank(dim: int, bits_per_bank: int, bit_offset: int = 0):
    bank = bitdrop_gpu.Bank()
    bank.bits_per_bank = bits_per_bank
    bank.bit_offset = bit_offset

    rules = []
    payload = []

    for i in range(bits_per_bank):
        r = bitdrop_gpu.Rule()
        r.type = bitdrop_gpu.RuleType.THRESHOLD
        r.bit = i
        r.dim = i % dim
        r.payload_offset = len(payload)

        thr = np.float16(0.0)
        payload.extend(np.frombuffer(thr.tobytes(), dtype=np.uint8).tolist())
        rules.append(r)

    bank.rules = rules
    bank.payload = payload
    return bank


# -------------------------------------------------------------------
# Embedding generator
# -------------------------------------------------------------------
def make_embeddings(num_vecs: int, dim: int, mode: str, seed: int | None):
    if seed is not None:
        np.random.seed(seed)

    if mode == "zero":
        return np.zeros((num_vecs, dim), dtype=np.float16)
    elif mode == "normal":
        return np.random.normal(0.0, 0.1, size=(num_vecs, dim)).astype(np.float16)
    elif mode == "uniform":
        return np.random.uniform(-0.5, 0.5, size=(num_vecs, dim)).astype(np.float16)
    else:
        # Fallback to zero for safety
        return np.zeros((num_vecs, dim), dtype=np.float16)


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
def log_result_csv(path: str, row: dict):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dim",
                "num_vecs",
                "bits_per_bank",
                "num_banks",
                "max_tags_per_vec",
                "chunk_size",
                "iters",
                "emb_mode",
                "profile",
                "throughput_vec_per_sec",
                "per_iter_sec",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# -------------------------------------------------------------------
# Main benchmark
# -------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="BitDrop GPU benchmark")
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--num-vecs", type=int, default=None)
    parser.add_argument("--bits-per-bank", type=int, default=None)
    parser.add_argument("--max-tags", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--emb-mode", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=None)
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args(argv)

    # Verbosity
    verbose = args.verbose if args.verbose is not None else get_env_int("BITDROP_VERBOSE", 1)
    def vprint(*a, **k):
        if verbose:
            print(*a, **k)

    # Profile presets
    profile = args.profile or get_env_str("BITDROP_PROFILE", "")
    dim_default = 1024
    num_vecs_default = 10_000
    chunk_default = 4096
    iters_default = 12

    if profile == "small":
        num_vecs_default = 2_000
        chunk_default = 2048
        iters_default = 20
    elif profile == "medium":
        num_vecs_default = 10_000
        chunk_default = 4096
        iters_default = 12
    elif profile == "large":
        num_vecs_default = 50_000
        chunk_default = 8192
        iters_default = 6

    DIM = args.dim or get_env_int("BITDROP_DIM", dim_default)
    NUM_VECS = args.num_vecs or get_env_int("BITDROP_NUM_VECS", num_vecs_default)
    BITS_PER_BANK = args.bits_per_bank or get_env_int("BITDROP_BITS_PER_BANK", 64)
    MAX_TAGS_PER_VEC = args.max_tags or get_env_int("BITDROP_MAX_TAGS_PER_VEC", 64)
    ITERS = args.iters or get_env_int("BITDROP_ITERS", iters_default)
    CHUNK_SIZE = args.chunk_size or get_env_int("BITDROP_CHUNK_SIZE", chunk_default)
    NUM_BANKS = get_env_int("BITDROP_NUM_BANKS", 1)
    EMB_MODE = args.emb_mode or get_env_str("BITDROP_EMB_MODE", "zero")
    LOG_FILE = args.log_file or get_env_str("BITDROP_LOG_FILE", "")

    SEED = get_env_int("BITDROP_SEED", 0)
    if SEED <= 0:
        SEED = None

    print("Loaded bitdrop_gpu from:", bitdrop_gpu.__file__)
    vprint(f"Profile={profile or 'none'}, DIM={DIM}, NUM_VECS={NUM_VECS}, "
           f"BITS_PER_BANK={BITS_PER_BANK}, NUM_BANKS={NUM_BANKS}, "
           f"MAX_TAGS_PER_VEC={MAX_TAGS_PER_VEC}, ITERS={ITERS}, "
           f"CHUNK_SIZE={CHUNK_SIZE}, EMB_MODE={EMB_MODE}, SEED={SEED}")

    # Embeddings (host) — BitDrop handles device side
    vprint(f"Creating embeddings: {NUM_VECS} x {DIM} ({EMB_MODE})")
    emb = make_embeddings(NUM_VECS, DIM, EMB_MODE, SEED)

    # Banks
    banks = []
    for b in range(NUM_BANKS):
        bit_offset = b * BITS_PER_BANK
        banks.append(build_threshold_bank(DIM, BITS_PER_BANK, bit_offset))

    num_rules = len(banks[0].rules) * NUM_BANKS
    total_bits = BITS_PER_BANK * NUM_BANKS

    vprint(f"Initializing BitDrop: dim={DIM}, num_rules={num_rules}, rule_bits={total_bits}")
    bitdrop_gpu.init(DIM, num_rules, total_bits)
    vprint("init: OK")

    # Internal GPU autotune
    try:
        vprint("Running internal BitDrop autotune...")
        bitdrop_gpu.autotune(DIM, num_rules)
        vprint("autotune: OK")
    except Exception as e:
        vprint("Internal autotune failed (non-fatal):", e)

    # Warmup
    WARMUP_ITERS = get_env_int("BITDROP_WARMUP_ITERS", 1)
    vprint(f"Warmup: {WARMUP_ITERS} iterations")
    for _ in range(WARMUP_ITERS):
        bitdrop_gpu.collapse_multi(
            emb,
            DIM,
            banks,
            total_bits,
            MAX_TAGS_PER_VEC,
            skim=True,
            skim_threshold=0.0,
            auto_chunk=True,
            chunk_size=CHUNK_SIZE,
        )

    # Timed loop
    start = time.perf_counter()
    for _ in range(ITERS):
        bits, tags, skim_mask = bitdrop_gpu.collapse_multi(
            emb,
            DIM,
            banks,
            total_bits,
            MAX_TAGS_PER_VEC,
            skim=True,
            skim_threshold=0.0,
            auto_chunk=True,
            chunk_size=CHUNK_SIZE,
        )
    end = time.perf_counter()

    per_iter = (end - start) / ITERS
    vprint(f"collapse_multi completed in {per_iter:.6f} sec")
    vprint(f"bits shape: {bits.shape}")
    vprint(f"tags shape: {tags.shape}")
    vprint(f"skim_mask shape: {skim_mask.shape}")

    vps = NUM_VECS / per_iter
    print(f"\nThroughput: {vps:,.0f} vectors/sec\n")

    # Optional logging
    if LOG_FILE:
        row = {
            "dim": DIM,
            "num_vecs": NUM_VECS,
            "bits_per_bank": BITS_PER_BANK,
            "num_banks": NUM_BANKS,
            "max_tags_per_vec": MAX_TAGS_PER_VEC,
            "chunk_size": CHUNK_SIZE,
            "iters": ITERS,
            "emb_mode": EMB_MODE,
            "profile": profile or "",
            "throughput_vec_per_sec": vps,
            "per_iter_sec": per_iter,
        }
        try:
            log_result_csv(LOG_FILE, row)
            vprint(f"Logged result to {LOG_FILE}")
        except Exception as e:
            vprint(f"Failed to log to {LOG_FILE}: {e}")

    bitdrop_gpu.shutdown()
    vprint("BitDrop shutdown complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Benchmark failed:", e, file=sys.stderr)
        sys.exit(1)










