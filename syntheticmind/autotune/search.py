"""
BitDrop Ultra Autotuner Search Space (v9‑Adaptive)
--------------------------------------------------
"""

def _valid(block, unroll, tile, rpt):
    if block % 32 != 0:
        return False

    warps = block // 32
    if warps < 2 or warps > 16:
        return False

    if unroll not in (1, 2):
        return False

    if tile < 8 or tile > 128:
        return False

    if rpt < 8 or rpt > 128:
        return False

    if tile * rpt > 4096:
        return False

    smem = tile * rpt * 4
    if smem > 48 * 1024:
        return False

    return True


def coarse_configs():
    block_space = [128, 192, 256, 384, 512]
    tile_space  = [16, 24, 32, 48, 64, 96]
    rpt_space   = [16, 24, 32, 48, 64]

    for block in block_space:
        for unroll in [1, 2]:
            for tile in tile_space:
                for rpt in rpt_space:
                    for multi_bank in [0, 1]:
                        if not _valid(block, unroll, tile, rpt):
                            continue

                        yield {
                            "stage": "coarse",
                            "block": block,
                            "unroll": unroll,
                            "tile": tile,
                            "rules_per_tile": rpt,
                            "multi_bank": multi_bank,
                            "similarity": 1,
                            "half2": 0,
                        }


def mid_configs(best):
    b   = best["block"]
    t   = best["tile"]
    rpt = best["rules_per_tile"]

    blocks = sorted(set([
        max(64, b // 2),
        b,
        min(1024, b * 2),
        b - 64,
        b + 64,
    ]))

    tiles = sorted(set([
        max(8, t // 2),
        t,
        min(128, t * 2),
        t - 16,
        t + 16,
    ]))

    rpts = sorted(set([
        max(8, rpt // 2),
        rpt,
        min(128, rpt * 2),
        rpt - 16,
        rpt + 16,
    ]))

    for block in blocks:
        for tile in tiles:
            for rules_per_tile in rpts:
                for unroll in [1, 2]:
                    if not _valid(block, unroll, tile, rules_per_tile):
                        continue

                    yield {
                        "stage": "mid",
                        "block": block,
                        "unroll": unroll,
                        "tile": tile,
                        "rules_per_tile": rules_per_tile,
                        "multi_bank": best["multi_bank"],
                        "similarity": 1,
                        "half2": 0,
                    }


def fine_configs(best):
    b   = best["block"]
    t   = best["tile"]
    rpt = best["rules_per_tile"]

    blocks = [b - 64, b - 32, b, b + 32, b + 64]
    tiles  = [t - 8, t - 4, t, t + 4, t + 8]
    rpts   = [rpt - 8, rpt - 4, rpt, rpt + 4, rpt + 8]

    for block in blocks:
        if block < 64 or block > 1024:
            continue

        for tile in tiles:
            if tile < 8 or tile > 128:
                continue

            for rules_per_tile in rpts:
                if rules_per_tile < 8 or rules_per_tile > 128:
                    continue

                for unroll in [1, 2]:
                    if not _valid(block, unroll, tile, rules_per_tile):
                        continue

                    yield {
                        "stage": "fine",
                        "block": block,
                        "unroll": unroll,
                        "tile": tile,
                        "rules_per_tile": rules_per_tile,
                        "multi_bank": best["multi_bank"],
                        "similarity": 1,
                        "half2": 0,
                    }


def generate_configs(best_from_cache=None):
    if best_from_cache is None:
        yield from coarse_configs()
    else:
        for cfg in mid_configs(best_from_cache):
            yield cfg
        for cfg in fine_configs(best_from_cache):
            yield cfg



