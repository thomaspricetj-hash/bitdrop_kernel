import json
import os
import tempfile
import shutil
import hashlib
import time

CACHE_FILE = r"C:\Users\thomas price\Desktop\kernal\syntheticmind\autotune_cache.json"
CACHE_VERSION = 4


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _compute_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_best():
    if not os.path.exists(CACHE_FILE):
        return None

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    required = {"config", "score", "version", "hash", "timestamp"}
    if not required.issubset(data.keys()):
        return None

    if not isinstance(data["config"], dict):
        return None
    if not isinstance(data["score"], (int, float)):
        return None

    if data["version"] != CACHE_VERSION:
        return None

    expected = _compute_hash({
        "config": data["config"],
        "score": data["score"],
        "version": data["version"],
        "timestamp": data["timestamp"],
    })
    if data["hash"] != expected:
        return None

    return data


def save_best(cfg: dict, score: float) -> None:
    _ensure_dir(CACHE_FILE)

    payload = {
        "version": CACHE_VERSION,
        "config": cfg,
        "score": score,
        "timestamp": time.time(),
    }

    payload["hash"] = _compute_hash(payload)

    tmp_fd, tmp_path = tempfile.mkstemp(prefix="autotune_", suffix=".json")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        shutil.move(tmp_path, CACHE_FILE)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise

