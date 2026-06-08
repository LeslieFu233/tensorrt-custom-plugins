cmake_minimum_required(VERSION 3.18)
project(tensor_custom_plugins LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 添加自定义 cmake 模块路径
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find CUDA
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA found: ${CUDA_VERSION}")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")
endif()

find_package(TensorRT REQUIRED)

add_library(tensorrt_custom_plugins_headers INTERFACE)
target_include_directories(tensorrt_custom_plugins_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_subdirectory(src/layernorm)
add_subdirectory(tests)
