cmake_minimum_required(VERSION 3.26)

project(bitdrop_gpu_python LANGUAGES CXX)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(bitdrop_gpu MODULE
    pybind_module.cpp
)

target_link_libraries(bitdrop_gpu
    PRIVATE
        bitdrop_gpu_core
        "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}/cudart_static.lib"
)

target_include_directories(bitdrop_gpu
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

set_target_properties(bitdrop_gpu PROPERTIES
    OUTPUT_NAME "bitdrop_gpu"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

if (WIN32)
    set_target_properties(bitdrop_gpu PROPERTIES SUFFIX ".pyd")
endif()

if (MSVC)
    target_compile_options(bitdrop_gpu PRIVATE /O2 /Ob2 /DNDEBUG)
endif()

set_target_properties(bitdrop_gpu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/python/Release"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/python/Release"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/python/Release"
)
