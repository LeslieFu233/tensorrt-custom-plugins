cmake_minimum_required(VERSION 3.18)
project(tensor_custom_plugins LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Find CUDA
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA found: ${CUDA_VERSION}")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")
endif()

# # Find TensorRT
# find_library(NVINFER_LIB nvinfer REQUIRED)
# find_library(NVINFER_PLUGIN_LIB nvinfer_plugin REQUIRED)
# find_path(TENSORRT_INCLUDE_DIR NvInfer.h REQUIRED)

# message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
# message(STATUS "TensorRT nvinfer:  ${NVINFER_LIB}")
# message(STATUS "TensorRT plugin:  ${NVINFER_PLUGIN_LIB}")

add_library(tensorrt_custom_plugins_headers INTERFACE)
target_include_directories(tensorrt_custom_plugins_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_subdirectory(src/layernorm)
add_subdirectory(tests)
