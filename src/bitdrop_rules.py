import struct
from typing import List, Dict, Tuple
import numpy as np

# ------------------------------------------------------------
# FP8 E4M3 encoding (raw, no scaling)
# ------------------------------------------------------------

USE_FP8_SIM_PAYLOAD = True  # FP8 similarity payloads ON by default


def fp8_e4m3_encode(vec_f16: np.ndarray) -> bytes:
    """
    Convert FP16 vector -> FP8 E4M3 (raw, no scaling).
    """
    assert vec_f16.dtype == np.float16
    f16 = vec_f16.view(np.uint16)

    sign = (f16 >> 15) & 0x1
    exp = (f16 >> 10) & 0x1F
    man = f16 & 0x3FF

    exp_fp8 = exp - 15 + 7
    exp_fp8 = np.clip(exp_fp8, 0, 0xF)

    man_fp8 = (man >> 7) & 0x7

    fp8 = (sign << 7) | (exp_fp8 << 3) | man_fp8
    return fp8.astype(np.uint8).tobytes()


def fp8_e4m3_encode_matrix(emb_f16: np.ndarray) -> np.ndarray:
    """
    Convert an embedding matrix [num_vecs, dim] from FP16 -> FP8 E4M3.
    Returns uint8 array of shape [num_vecs, dim].
    """
    assert emb_f16.dtype == np.float16

    f16 = emb_f16.view(np.uint16)

    sign = (f16 >> 15) & 0x1
    exp = (f16 >> 10) & 0x1F
    man = f16 & 0x3FF

    exp_fp8 = exp - 15 + 7
    exp_fp8 = np.clip(exp_fp8, 0, 0xF)

    man_fp8 = (man >> 7) & 0x7

    fp8 = (sign << 7) | (exp_fp8 << 3) | man_fp8
    return fp8.astype(np.uint8)

# ------------------------------------------------------------
# BitDropRule layout
# ------------------------------------------------------------

# type: uint8
# bit:  uint8
# dim:  uint16
# payload_offset: uint32
BITDROP_RULE_STRUCT = struct.Struct("<BBHI")

BITDROP_RULE_THRESHOLD = 0
BITDROP_RULE_SIMILARITY = 1

# ------------------------------------------------------------
# Payload builders
# ------------------------------------------------------------


def _append_threshold_payload(payload: bytearray, threshold: float, tag_id: int) -> int:
    """
    Layout:
        [0:2]  threshold (fp16)
        [2:4]  reserved/pad (fp16)
        [4:6]  tag_id (uint16, packed)
    """
    offset = len(payload)

    thr_h = np.float16(threshold)
    pad_h = np.float16(0.0)

    payload += thr_h.tobytes()
    payload += pad_h.tobytes()

    tag_u16 = np.uint16(tag_id & 0xFFFF)
    payload += tag_u16.tobytes()

    return offset


def _append_similarity_payload(payload: bytearray, vec: np.ndarray, threshold: float, tag_id: int) -> int:
    """
    Layout (FP8 mode):
        [0:dim]        similarity vector (fp8 bytes)
        [dim:dim+2]    threshold (fp16)
        [dim+2:dim+4]  pad (fp16)
        [dim+4:dim+6]  tag_id (uint16, packed)
    Layout (FP16 mode):
        [0:2*dim]          similarity vector (fp16)
        [2*dim:2*dim+2]    threshold (fp16)
        [2*dim+2:2*dim+4]  pad (fp16)
        [2*dim+4:2*dim+6]  tag_id (uint16, packed)
    """
    assert vec.dtype == np.float16
    dim = int(vec.shape[0])

    offset = len(payload)

    if USE_FP8_SIM_PAYLOAD:
        payload += fp8_e4m3_encode(vec)
    else:
        payload += vec.tobytes()

    thr_h = np.float16(threshold)
    pad_h = np.float16(0.0)
    payload += thr_h.tobytes()
    payload += pad_h.tobytes()

    tag_u16 = np.uint16(tag_id & 0xFFFF)
    payload += tag_u16.tobytes()

    return offset

# ------------------------------------------------------------
# Rule constructors
# ------------------------------------------------------------


def make_threshold_rule(bit: int,
                        dim_idx: int,
                        threshold: float,
                        tag_id: int,
                        payload: bytearray) -> bytes:
    payload_offset = _append_threshold_payload(payload, threshold, tag_id)
    return BITDROP_RULE_STRUCT.pack(
        BITDROP_RULE_THRESHOLD,
        bit & 0xFF,
        dim_idx & 0xFFFF,
        payload_offset,
    )


def make_similarity_rule(bit: int,
                         vec: np.ndarray,
                         threshold: float,
                         tag_id: int,
                         payload: bytearray) -> bytes:
    assert vec.dtype == np.float16
    dim = int(vec.shape[0])

    payload_offset = _append_similarity_payload(payload, vec, threshold, tag_id)

    return BITDROP_RULE_STRUCT.pack(
        BITDROP_RULE_SIMILARITY,
        bit & 0xFF,
        dim & 0xFFFF,
        payload_offset,
    )

# ------------------------------------------------------------
# Bank builder + merging
# ------------------------------------------------------------


def build_bank(rules: List[bytes],
               payload: bytearray,
               bits_per_bank: int) -> Dict:
    if bits_per_bank != 64:
        raise ValueError("bits_per_bank must be exactly 64")

    return {
        "rules": b"".join(rules),
        "payload": bytes(payload),
        "bits": int(bits_per_bank),
    }


def merge_banks(banks: List[Dict]) -> Dict:
    """
    Merge multiple banks into a single bank with adjusted bit offsets.
    Assumes each bank has 'bits' == 64 and non-overlapping bit ranges.
    """
    merged_rules = bytearray()
    merged_payload = bytearray()
    bit_offset = 0
    payload_offset = 0

    for bank in banks:
        rules_blob = bank["rules"]
        payload_blob = bank["payload"]
        bits = bank["bits"]

        num_rules = len(rules_blob) // BITDROP_RULE_STRUCT.size
        for i in range(num_rules):
            r_bytes = rules_blob[i * BITDROP_RULE_STRUCT.size:(i + 1) * BITDROP_RULE_STRUCT.size]
            r_type, r_bit, r_dim, r_po = BITDROP_RULE_STRUCT.unpack(r_bytes)

            new_bit = r_bit + bit_offset
            new_po = r_po + payload_offset

            merged_rules += BITDROP_RULE_STRUCT.pack(
                r_type,
                new_bit & 0xFF,
                r_dim & 0xFFFF,
                new_po,
            )

        merged_payload += payload_blob
        bit_offset += bits
        payload_offset += len(payload_blob)

    return {
        "rules": bytes(merged_rules),
        "payload": bytes(merged_payload),
        "bits": bit_offset,
    }

# ------------------------------------------------------------
# Simple test banks
# ------------------------------------------------------------


def build_simple_threshold_bank(dim: int,
                                num_rules: int,
                                bits_per_bank: int = 64) -> Dict:
    payload = bytearray()
    rules: List[bytes] = []
    for i in range(num_rules):
        rules.append(
            make_threshold_rule(
                i % bits_per_bank,
                i % dim,
                0.0,
                i,
                payload,
            )
        )
    return build_bank(rules, payload, bits_per_bank)


def build_simple_similarity_bank(dim: int,
                                 num_rules: int,
                                 bits_per_bank: int = 64) -> Dict:
    payload = bytearray()
    rules: List[bytes] = []
    for i in range(num_rules):
        vec = np.random.randn(dim).astype(np.float16)
        rules.append(
            make_similarity_rule(
                i % bits_per_bank,
                vec,
                0.0,
                i,
                payload,
            )
        )
    return build_bank(rules, payload, bits_per_bank)




