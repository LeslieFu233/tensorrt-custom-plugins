#ifndef TENSORRT_CUSTOM_PLUGIN_UTILS_H
#define TENSORRT_CUSTOM_PLUGIN_UTILS_H

#include <iostream>

namespace tensorrt_custom_plugins {
namespace utils{
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define CUDA_CHECK(val)                                                   \
    do {                                                                  \
        cudaError_t err = (val);                                          \
                                                                          \
        if (err != cudaSuccess) {                                         \
            std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err)       \
                      << " | file=" << __FILE__ << " | line=" << __LINE__ \
                      << " | expr=" << #val << std::endl;                 \
                                                                          \
            std::abort();                                                 \
        }                                                                 \
    } while (0)
}
}  // namespace tensorrt_custom_plugins

#endif
