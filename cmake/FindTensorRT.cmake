# ---------------------------------------------------------------
# FindTensorRT.cmake
# 用于 TensorRT 10.x tar 包
# 用法:
#   list(APPEND CMAKE_MODULE_PATH "/path/to/cmake")
#   find_package(TensorRT REQUIRED)
# 提供:
#   TensorRT_INCLUDE_DIRS
#   TensorRT_LIBRARIES (nvinfer + nvinfer_plugin + nvonnxparser)
# ---------------------------------------------------------------

# 尝试用环境变量
if(NOT TENSORRT_ROOT)
    set(TENSORRT_ROOT $ENV{TENSORRT_HOME})
endif()
# 查找 include
find_path(TensorRT_INCLUDE_DIR
    NvInfer.h
    HINTS
        ${TENSORRT_ROOT}/include
        /usr/include
        /usr/local/include
)

# 查找主要库
find_library(TensorRT_LIBRARY
    nvinfer
    HINTS
        ${TENSORRT_ROOT}/lib
        /usr/lib
        /usr/local/lib
)

# 查找 plugin 库
find_library(TensorRT_PLUGIN_LIBRARY
    nvinfer_plugin
    HINTS
        ${TENSORRT_ROOT}/lib
)

# 查找 ONNX parser 库
find_library(TensorRT_ONNXPARSER_LIBRARY
    nvonnxparser
    HINTS
        ${TENSORRT_ROOT}/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    TensorRT
    REQUIRED_VARS
        TensorRT_INCLUDE_DIR
        TensorRT_LIBRARY
)

# 最终提供给 target_link_libraries / include_directories 使用
set(TensorRT_LIBRARIES
    ${TensorRT_LIBRARY}
    ${TensorRT_PLUGIN_LIBRARY}
    ${TensorRT_ONNXPARSER_LIBRARY}
)

set(TensorRT_INCLUDE_DIRS
    ${TensorRT_INCLUDE_DIR}
)
