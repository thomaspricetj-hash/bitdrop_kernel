import subprocess
import os
import hashlib
import json
import shutil
import tempfile
import re
import time

BUILD_DIR = r"C:\Users\thomas price\Desktop\kernal\build"
SRC_DIR   = r"C:\Users\thomas price\Desktop\kernal"
HASH_FILE = os.path.join(BUILD_DIR, "bitdrop_build_hash.json")
METRICS_FILE = os.path.join(BUILD_DIR, "bitdrop_build_metrics.json")

PTXAS_RE = re.compile(
    r"ptxas info\s*: Used (\d+) registers, (\d+) bytes smem, "
    r"(\d+) bytes cmem, (\d+) bytes lmem",
    re.IGNORECASE
)


def _run(cmd, cwd):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
    )

    stdout = proc.stdout
    stderr = proc.stderr
    combined = stdout + "\n" + stderr

    metrics = _extract_ptxas_metrics(combined)
    if metrics:
        _save_metrics(metrics)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n"
            f"  {' '.join(cmd)}\n\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}\n"
        )

    return stdout


def _extract_ptxas_metrics(text):
    matches = PTXAS_RE.findall(text)
    if not matches:
        return None

    kernels = []
    for regs, smem, cmem, lmem in matches:
        regs = int(regs)
        smem = int(smem)
        cmem = int(cmem)
        lmem = int(lmem)

        kernels.append({
            "registers": regs,
            "shared_mem": smem,
            "const_mem": cmem,
            "local_mem": lmem,
            "spill_bytes": lmem,
            "occupancy_risk": _occupancy_risk(regs),
            "smem_pressure": _smem_pressure(smem),
        })

    return {
        "timestamp": time.time(),
        "kernels": kernels,
        "kernel_count": len(kernels),
    }


def _occupancy_risk(regs):
    if regs < 64:
        return "low"
    if regs <= 96:
        return "medium"
    return "high"


def _smem_pressure(smem):
    if smem < 16 * 1024:
        return "low"
    if smem <= 32 * 1024:
        return "medium"
    return "high"


def _save_metrics(metrics):
    os.makedirs(BUILD_DIR, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="buildmetrics_", suffix=".json")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        shutil.move(tmp_path, METRICS_FILE)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def _compute_kernel_hash():
    h = hashlib.sha256()

    def _hash_file(path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                h.update(f.read())

    _hash_file(os.path.join(SRC_DIR, "src", "bitdrop_gpu.cu"))
    _hash_file(os.path.join(SRC_DIR, "python", "pybind_module.cpp"))
    _hash_file(os.path.join(SRC_DIR, "CMakeLists.txt"))

    headers_dir = os.path.join(SRC_DIR, "src")
    if os.path.exists(headers_dir):
        for fname in os.listdir(headers_dir):
            if fname.endswith((".h", ".hpp")):
                _hash_file(os.path.join(headers_dir, fname))

    return h.hexdigest()


def _load_previous_hash():
    if not os.path.exists(HASH_FILE):
        return None
    try:
        with open(HASH_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("hash")
    except Exception:
        return None


def _save_hash(h):
    os.makedirs(BUILD_DIR, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="buildhash_", suffix=".json")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump({"hash": h}, f, indent=2)
        shutil.move(tmp_path, HASH_FILE)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def rebuild_pyd() -> bool:
    os.makedirs(BUILD_DIR, exist_ok=True)

    new_hash = _compute_kernel_hash()
    old_hash = _load_previous_hash()

    metrics_missing = not os.path.exists(METRICS_FILE)

    if old_hash == new_hash and not metrics_missing:
        print("[build] No changes detected — skipping rebuild.")
        return True

    print("[build] Changes detected — rebuilding bitdrop_gpu...")

    cmake_cache = os.path.join(BUILD_DIR, "CMakeCache.txt")
    if not os.path.exists(cmake_cache):
        print("[build] Running CMake configure...")
        _run(["cmake", SRC_DIR], cwd=BUILD_DIR)

    print("[build] Building bitdrop_gpu...")
    _run([
        "cmake",
        "--build", BUILD_DIR,
        "--config", "Release",
        "--target", "bitdrop_gpu"
    ], cwd=SRC_DIR)

    _save_hash(new_hash)

    print("[build] Build successful.")
    return True

