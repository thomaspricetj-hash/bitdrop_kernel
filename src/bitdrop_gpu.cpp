#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "bitdrop_gpu.h"

namespace py = pybind11;

// ------------------------------------------------------------
//  Helpers
// ------------------------------------------------------------

static void check(bool cond, const char* msg) {
    if (!cond) throw std::runtime_error(msg);
}

static BitDropStatus check_status(BitDropStatus st, const char* msg) {
    if (st != BITDROP_OK) {
        throw std::runtime_error(msg);
    }
    return st;
}

// ------------------------------------------------------------
//  Convert Python banks → BitDropBank[]
// ------------------------------------------------------------

static std::vector<BitDropBank>
build_banks(const py::list& banks,
            std::vector<std::vector<BitDropRule>>& rule_storage,
            std::vector<std::vector<std::uint8_t>>& payload_storage,
            std::vector<int>& bits_per_bank,
            std::vector<int>& bit_offsets)
{
    int num_banks = (int)banks.size();
    check(num_banks > 0, "banks list is empty");

    rule_storage.resize(num_banks);
    payload_storage.resize(num_banks);
    bits_per_bank.resize(num_banks);
    bit_offsets.resize(num_banks);

    // Compute bit offsets
    int offset = 0;
    for (int i = 0; i < num_banks; ++i) {
        auto bank = banks[i].cast<py::dict>();

        bits_per_bank[i] = bank["bits"].cast<int>();
        // Enforce 64-bit banks and byte alignment
        check(bits_per_bank[i] == 64,
              "bits_per_bank must be exactly 64");
        check(bits_per_bank[i] % 8 == 0,
              "bits_per_bank must be a multiple of 8");

        bit_offsets[i] = offset;
        offset += bits_per_bank[i];
    }

    // Copy rule + payload buffers
    for (int i = 0; i < num_banks; ++i) {
        auto bank = banks[i].cast<py::dict>();

        // rules: Python bytes → vector<BitDropRule>
        py::bytes rules_py = bank["rules"].cast<py::bytes>();
        std::string rules_str = rules_py;
        check(rules_str.size() % sizeof(BitDropRule) == 0,
              "rules buffer size invalid");

        std::size_t num_rules = rules_str.size() / sizeof(BitDropRule);
        rule_storage[i].resize(num_rules);
        std::memcpy(rule_storage[i].data(), rules_str.data(), rules_str.size());

        // payload: Python bytes → vector<uint8_t>
        py::bytes payload_py = bank["payload"].cast<py::bytes>();
        std::string payload_str = payload_py;
        payload_storage[i].resize(payload_str.size());
        if (!payload_str.empty()) {
            std::memcpy(payload_storage[i].data(),
                        payload_str.data(),
                        payload_str.size());
        }
    }

    // Build BitDropBank[]
    std::vector<BitDropBank> out(num_banks);
    for (int i = 0; i < num_banks; ++i) {
        out[i].rules         = rule_storage[i].data();
        out[i].payload       = payload_storage[i].data();
        out[i].num_rules     = (int)rule_storage[i].size();
        out[i].bits_per_bank = bits_per_bank[i];
        out[i].bit_offset    = bit_offsets[i];
        out[i].payload_bytes = (int)payload_storage[i].size();
    }

    return out;
}

// ------------------------------------------------------------
//  collapse_multi wrapper
// ------------------------------------------------------------

py::dict collapse_multi_wrapper(
    py::array_t<std::uint16_t, py::array::c_style | py::array::forcecast> emb,
    py::list banks,
    bool return_tags = true,
    bool skim = false,
    float skim_threshold = 0.0f,
    bool auto_chunk = true,
    int chunk_size = 4096)
{
    // Validate embeddings
    check(emb.ndim() == 2, "embeddings must be 2D");
    int num_vecs = (int)emb.shape(0);
    int dim      = (int)emb.shape(1);

    const void* emb_ptr = emb.data();

    // Build banks
    std::vector<std::vector<BitDropRule>>   rule_storage;
    std::vector<std::vector<std::uint8_t>>  payload_storage;
    std::vector<int>                        bits_per_bank;
    std::vector<int>                        bit_offsets;

    auto banks_vec = build_banks(
        banks, rule_storage, payload_storage,
        bits_per_bank, bit_offsets);

    int num_banks = (int)banks_vec.size();

    // Compute total bits
    int total_bits = 0;
    for (int b : bits_per_bank) total_bits += b;
    check(total_bits > 0, "total_bits must be > 0");
    check(total_bits % 8 == 0, "total_bits must be a multiple of 8");
    check(total_bits % 64 == 0, "total_bits must be a multiple of 64");
    int total_bytes = total_bits / 8;

    // Auto-tag-capacity
    int max_possible_tags = 0;
    for (auto& bank : banks_vec) {
        max_possible_tags += bank.num_rules;
    }
    int max_tags_per_vec = return_tags ? std::min(max_possible_tags, 32) : 0;

    // Allocate output buffers
    py::array_t<std::uint64_t> bits_out({num_vecs, num_banks});
    py::array_t<std::int32_t>  tags_out;
    py::array_t<std::uint8_t>  skim_mask_out;

    if (return_tags) {
        tags_out = py::array_t<std::int32_t>({num_vecs, max_tags_per_vec});
    }
    if (skim) {
        skim_mask_out = py::array_t<std::uint8_t>({num_vecs});
    }

    // Flatten bits_out to raw byte buffer
    std::vector<std::uint8_t> bits_flat(
        (std::size_t)num_vecs * (std::size_t)total_bytes, 0);

    BitDropStatus st;
    {
        // Release GIL during GPU work
        py::gil_scoped_release release;

        st = bitdrop_collapse_multi(
            emb_ptr,
            num_vecs,
            dim,
            banks_vec.data(),
            num_banks,
            total_bits,
            bits_flat.data(),
            return_tags ? tags_out.mutable_data() : nullptr,
            max_tags_per_vec,
            skim ? skim_mask_out.mutable_data() : nullptr,
            skim ? 1 : 0,
            skim_threshold,
            auto_chunk ? 1 : 0,
            chunk_size);
    }

    check_status(st, "bitdrop_collapse_multi failed");

    // Convert flat bits → uint64 per bank
    auto bits_view = bits_out.mutable_unchecked<2>();
    for (int i = 0; i < num_vecs; ++i) {
        for (int b = 0; b < num_banks; ++b) {
            int offset = bit_offsets[b] / 8;
            std::uint64_t v = 0;
            std::memcpy(&v,
                        &bits_flat[(std::size_t)i * (std::size_t)total_bytes + offset],
                        sizeof(std::uint64_t));
            bits_view(i, b) = v;
        }
    }

    // Build return dict
    py::dict out;
    out["bits"] = bits_out;

    if (return_tags) out["tags"] = tags_out;
    if (skim)       out["skim_mask"] = skim_mask_out;

    // Bank metadata
    py::list meta;
    for (int i = 0; i < num_banks; ++i) {
        py::dict m;
        m["bits"]   = bits_per_bank[i];
        m["offset"] = bit_offsets[i];
        m["rules"]  = (int)rule_storage[i].size();
        meta.append(m);
    }
    out["bank_meta"] = meta;

    return out;
}

// ------------------------------------------------------------
//  PYBIND11 MODULE
// ------------------------------------------------------------

PYBIND11_MODULE(bitdrop_gpu, m) {
    m.doc() = "BitDrop GPU multi-bank collapse module";

    m.def("init",
          [](int dim, int num_rules, int rule_bits) {
              check_status(bitdrop_init(dim, num_rules, rule_bits),
                           "bitdrop_init failed");
          },
          py::arg("dim"),
          py::arg("num_rules"),
          py::arg("rule_bits"));

    m.def("autotune",
          [](int dim, int num_rules) {
              check_status(bitdrop_autotune(dim, num_rules),
                           "bitdrop_autotune failed");
          },
          py::arg("dim"),
          py::arg("num_rules"));

    m.def("collapse_multi", &collapse_multi_wrapper,
          py::arg("emb"),
          py::arg("banks"),
          py::arg("return_tags") = true,
          py::arg("skim") = false,
          py::arg("skim_threshold") = 0.0f,
          py::arg("auto_chunk") = true,
          py::arg("chunk_size") = 4096);

    m.def("get_tuning", []() {
        BitDropTuning t = bitdrop_get_tuning();
        py::dict d;
        d["block_size"]       = t.block_size;
        d["rules_per_tile"]   = t.rules_per_tile;
        d["warps_per_vector"] = t.warps_per_vector;
        d["unroll"]           = t.unroll;
        return d;
    });

    m.def("shutdown", []() {
        bitdrop_shutdown();
    });
}

