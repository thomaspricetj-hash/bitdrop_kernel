cmake_minimum_required(VERSION 3.26)
project(bitdrop_gpu LANGUAGES C CXX CUDA)

# ------------------------------------------------------------
# Global settings
# ------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ------------------------------------------------------------
# CUDA architecture (CUDA 13.3 supports sm_90)
# ------------------------------------------------------------
set(CMAKE_CUDA_ARCHITECTURES 90)

# ------------------------------------------------------------
# Core GPU library
# ------------------------------------------------------------
# IMPORTANT:
#   Only compile bitdrop_gpu.cu — it contains ALL CUDA + host code.
add_library(bitdrop_gpu_core STATIC
    src/bitdrop_gpu.cu
)

target_include_directories(bitdrop_gpu_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ------------------------------------------------------------
# FIX: Disable separable compilation (NO RDC)
# ------------------------------------------------------------
set_target_properties(bitdrop_gpu_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)

# ------------------------------------------------------------
# Link CUDA runtime manually (CUDA 13.x)
# ------------------------------------------------------------
target_link_libraries(bitdrop_gpu_core
    PRIVATE
        "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}/cudart_static.lib"
)

# ------------------------------------------------------------
# Python bindings
# ------------------------------------------------------------
option(BITDROP_BUILD_PYTHON "Build Python bindings" ON)

if (BITDROP_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    add_subdirectory(python)
endif()
