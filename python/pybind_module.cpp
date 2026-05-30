#pragma message(">>> USING CLEAN pybind_module.cpp WITH DEBUG <<<")

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cuda_fp16.h>
#include "bitdrop_gpu.h"

namespace py = pybind11;

// ------------------------------------------------------------
// Convert Python NumPy array of float16 → raw pointer (__half)
// ------------------------------------------------------------
static const __half* as_half_ptr(const py::array& arr) {
    if (arr.dtype().kind() != 'f' || arr.dtype().itemsize() != 2)
        throw std::runtime_error("Expected float16 array");
    return reinterpret_cast<const __half*>(arr.data());
}

// ------------------------------------------------------------
// Convert Python bytes → raw uint8_t*
// ------------------------------------------------------------
static const std::uint8_t* as_u8_ptr(const py::bytes& b) {
    std::string tmp = b;
    return reinterpret_cast<const std::uint8_t*>(tmp.data());
}

// ------------------------------------------------------------
// Python wrapper for BitDropBank
// ------------------------------------------------------------
struct PyBitDropBank {
    std::vector<BitDropRule>  rules;
    std::vector<std::uint8_t> payload;
    int bits_per_bank = 64;
    int bit_offset    = 0;

    BitDropBank to_c_struct() const {
        BitDropBank b{};
        b.rules         = rules.data();
        b.payload       = payload.data();
        b.num_rules     = static_cast<int>(rules.size());
        b.bits_per_bank = bits_per_bank;
        b.bit_offset    = bit_offset;
        b.payload_bytes = static_cast<int>(payload.size());
        return b;
    }
};

// ------------------------------------------------------------
// PYBIND11 MODULE
// ------------------------------------------------------------
PYBIND11_MODULE(bitdrop_gpu, m) {
    m.doc() = "BitDrop GPU Python bindings (full P2 interface)";

    // Status enum
    py::enum_<BitDropStatus>(m, "Status")
        .value("OK",              BITDROP_OK)
        .value("ERR_INVALID_ARG", BITDROP_ERR_INVALID_ARG)
        .value("ERR_NOT_INIT",    BITDROP_ERR_NOT_INIT)
        .value("ERR_CUDA",        BITDROP_ERR_CUDA)
        .value("ERR_INTERNAL",    BITDROP_ERR_INTERNAL);

    // Rule type enum
    py::enum_<BitDropRuleType>(m, "RuleType")
        .value("THRESHOLD",  BITDROP_RULE_THRESHOLD)
        .value("SIMILARITY", BITDROP_RULE_SIMILARITY);

    // BitDropRule struct
    py::class_<BitDropRule>(m, "Rule")
        .def(py::init<>())
        .def_readwrite("type",           &BitDropRule::type)
        .def_readwrite("bit",            &BitDropRule::bit)
        .def_readwrite("dim",            &BitDropRule::dim)
        .def_readwrite("payload_offset", &BitDropRule::payload_offset);

    // PyBitDropBank wrapper
    py::class_<PyBitDropBank>(m, "Bank")
        .def(py::init<>())
        .def_readwrite("rules",         &PyBitDropBank::rules)
        .def_readwrite("payload",       &PyBitDropBank::payload)
        .def_readwrite("bits_per_bank", &PyBitDropBank::bits_per_bank)
        .def_readwrite("bit_offset",    &PyBitDropBank::bit_offset)
        .def("to_c", &PyBitDropBank::to_c_struct);

    // Tuning struct
    py::class_<BitDropTuning>(m, "Tuning")
        .def_readwrite("block_size",       &BitDropTuning::block_size)
        .def_readwrite("rules_per_tile",   &BitDropTuning::rules_per_tile)
        .def_readwrite("warps_per_vector", &BitDropTuning::warps_per_vector)
        .def_readwrite("unroll",           &BitDropTuning::unroll);

    // Core API
    m.def("init", [](int dim, int num_rules, int rule_bits) {
        return bitdrop_init(dim, num_rules, rule_bits);
    });

    m.def("autotune", [](int dim, int num_rules) {
        return bitdrop_autotune(dim, num_rules);
    });

    m.def("shutdown", []() {
        bitdrop_shutdown();
    });

    m.def("set_similarity_chunk", [](int chunk) {
        bitdrop_set_similarity_chunk(chunk);
    });

    m.def("get_tuning", []() {
        return bitdrop_get_tuning();
    });

    // ------------------------------------------------------------
    // collapse_multi WITH DEBUG
    // ------------------------------------------------------------
    m.def("collapse_multi",
        [](py::array embeddings_f16,
           int dim,
           std::vector<PyBitDropBank> banks,
           int total_bits,
           int max_tags_per_vec,
           bool skim,
           float skim_threshold,
           bool auto_chunk,
           int chunk_size)
        {
            int num_vecs = static_cast<int>(embeddings_f16.shape(0));

            std::vector<BitDropBank> c_banks;
            c_banks.reserve(banks.size());
            for (auto& b : banks)
                c_banks.push_back(b.to_c_struct());

            int total_bytes = total_bits / 8;

            py::array_t<std::uint8_t> out_bits({num_vecs, total_bytes});
            py::array_t<std::int32_t> out_tags({num_vecs, max_tags_per_vec});
            py::array_t<std::uint8_t> skim_mask({num_vecs});

            BitDropStatus st = bitdrop_collapse_multi(
                as_half_ptr(embeddings_f16),
                num_vecs,
                dim,
                c_banks.data(),
                static_cast<int>(c_banks.size()),
                total_bits,
                out_bits.mutable_data(),
                out_tags.mutable_data(),
                max_tags_per_vec,
                skim ? skim_mask.mutable_data() : nullptr,
                skim ? 1 : 0,
                skim_threshold,
                auto_chunk ? 1 : 0,
                chunk_size
            );

            if (st != BITDROP_OK) {
                std::stringstream ss;
                ss << "collapse_multi failed with status: " << st;

                // If CUDA error, print it
                if (st == BITDROP_ERR_CUDA) {
                    ss << " (CUDA error: " << cudaGetErrorString(cudaGetLastError()) << ")";
                }

                throw std::runtime_error(ss.str());
            }

            return py::make_tuple(out_bits, out_tags, skim_mask);
        },
        py::arg("embeddings_f16"),
        py::arg("dim"),
        py::arg("banks"),
        py::arg("total_bits"),
        py::arg("max_tags_per_vec"),
        py::arg("skim") = false,
        py::arg("skim_threshold") = 0.0f,
        py::arg("auto_chunk") = true,
        py::arg("chunk_size") = 0
    );
}


