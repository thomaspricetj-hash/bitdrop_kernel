cmake_minimum_required(VERSION 3.26)
project(BitDropGPU LANGUAGES C CXX CUDA)

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
# CUDA architecture selection
# ------------------------------------------------------------
if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.4)
    message(STATUS "CUDA < 12.4 detected → using sm_89")
    set(CMAKE_CUDA_ARCHITECTURES 89)
else()
    message(STATUS "CUDA >= 12.4 detected → using sm_90")
    set(CMAKE_CUDA_ARCHITECTURES 90)
endif()

# ------------------------------------------------------------
# Subprojects
# ------------------------------------------------------------
add_subdirectory(src/bitdrop_gpu)
add_subdirectory(python)

# ------------------------------------------------------------
# BitDrop GPU Benchmark
# ------------------------------------------------------------
add_executable(bitdrop_bench
    src/bench_bitdrop_gpu.cpp
)

target_link_libraries(bitdrop_bench
    PRIVATE
        bitdrop_gpu_core
        "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}/cudart_static.lib"
)

target_include_directories(bitdrop_bench
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)



