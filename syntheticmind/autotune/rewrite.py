import re
import os

# FIXED: point to the actual kernel file used by your build system
KERNEL_FILE = r"C:\Users\thomas price\Desktop\kernal\src\bitdrop_gpu\bitdrop_kernel_main.cu"

DEFINE_PATTERN = r"^\s*#\s*define\s+{name}\s+\S+"


def _set_define(src: str, name: str, value) -> str:
    if isinstance(value, bool):
        value = int(value)
    elif isinstance(value, float):
        value = int(value)
    elif isinstance(value, str):
        pass
    else:
        value = int(value)

    pattern = DEFINE_PATTERN.format(name=re.escape(name))
    repl = f"#define {name} {value}"

    lines = src.splitlines()
    new_lines = []
    replaced = False

    for line in lines:
        if re.match(pattern, line):
            if not replaced:
                new_lines.append(repl)
                replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        insert_at = 0
        for i, line in enumerate(new_lines):
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("/*"):
                continue
            insert_at = i
            break
        new_lines.insert(insert_at, repl)

    return "\n".join(new_lines)


def _clamp_cfg(cfg):
    c = dict(cfg)

    b = max(32, int(c["block"]))
    if b % 32 != 0:
        b = (b // 32) * 32
        if b < 32:
            b = 32
    c["block"] = b

    u = int(c["unroll"])
    if u not in (1, 2):
        u = 1 if u <= 1 else 2
    c["unroll"] = u

    t = max(8, min(128, int(c["tile"])))
    c["tile"] = t

    rpt = max(8, min(128, int(c["rules_per_tile"])))
    c["rules_per_tile"] = rpt

    c["multi_bank"] = 1 if int(c["multi_bank"]) else 0
    c["similarity"] = 1
    c["half2"] = 0

    return c


def rewrite_kernel(cfg: dict) -> None:
    if not os.path.exists(KERNEL_FILE):
        raise FileNotFoundError(f"Kernel file not found: {KERNEL_FILE}")

    cfg = _clamp_cfg(cfg)

    with open(KERNEL_FILE, "r", encoding="utf-8") as f:
        src = f.read()

    src = _set_define(src, "BLOCK_SIZE",        cfg["block"])
    src = _set_define(src, "UNROLL",            cfg["unroll"])
    src = _set_define(src, "TILE",              cfg["tile"])
    src = _set_define(src, "RULES_PER_TILE",    cfg["rules_per_tile"])
    src = _set_define(src, "MULTI_BANK",        cfg["multi_bank"])
    src = _set_define(src, "ENABLE_SIMILARITY", cfg["similarity"])
    src = _set_define(src, "USE_HALF2",         cfg["half2"])

    tmp_path = KERNEL_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(src)
    os.replace(tmp_path, KERNEL_FILE)

    print(f"[rewrite] Updated kernel defines (clamped): {cfg}")




