cmake_minimum_required(VERSION 3.26)

project(bitdrop_gpu_core LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ------------------------------------------------------------
# Force static CUDA runtime
# ------------------------------------------------------------
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

# ------------------------------------------------------------
# CUDA architecture (RTX 4090 = SM 90)
# ------------------------------------------------------------
set(CMAKE_CUDA_ARCHITECTURES 90)

# ------------------------------------------------------------
# Source files
# ------------------------------------------------------------
set(BITDROP_GPU_SOURCES
    bitdrop_kernel_main.cu
    bitdrop_gpu_host.cpp
)

set(BITDROP_GPU_HEADERS
    bitdrop_kernel_activity.cuh
    bitdrop_kernel_similarity.cuh
    bitdrop_kernel_wavefront.cuh
    bitdrop_kernel_refine.cuh
    bitdrop_kernel_locality.cuh
)

# ------------------------------------------------------------
# Compile host TU with NVCC
# ------------------------------------------------------------
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/bitdrop_gpu_host.cpp
    PROPERTIES LANGUAGE CUDA
)

# ------------------------------------------------------------
# Build static library
# ------------------------------------------------------------
add_library(bitdrop_gpu_core STATIC
    ${BITDROP_GPU_SOURCES}
    ${BITDROP_GPU_HEADERS}
)

target_include_directories(bitdrop_gpu_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# ------------------------------------------------------------
# IMPORTANT: Disable separable compilation (removes RDC + device registration)
# ------------------------------------------------------------
set_target_properties(bitdrop_gpu_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)

# ------------------------------------------------------------
# Link CUDA runtime
# ------------------------------------------------------------
target_link_libraries(bitdrop_gpu_core
    PRIVATE
        "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}/cudart_static.lib"
)

# ------------------------------------------------------------
# Windows-specific flags
# ------------------------------------------------------------
if (MSVC)
    target_compile_options(bitdrop_gpu_core PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W1 /EHsc /bigobj>
    )
endif()

# ------------------------------------------------------------
# CUDA: enable device-side lambdas
# ------------------------------------------------------------
target_compile_options(bitdrop_gpu_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

set(BITDROP_GPU_CORE_LIB bitdrop_gpu_core PARENT_SCOPE)







