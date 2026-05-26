#ifndef TENSORRT_CUSTOM_PLUGIN_UTILS_H
#define TENSORRT_CUSTOM_PLUGIN_UTILS_H

#include <chrono>
#include <iostream>

namespace tensorrt_custom_plugins {
namespace utils {

class ScopedTimer {
public:
    ScopedTimer(float& out_ms)
        : out_ms_(out_ms), start_(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        out_ms_ =
            std::chrono::duration<float, std::milli>(end - start_).count();
    }

private:
    float& out_ms_;
    std::chrono::high_resolution_clock::time_point start_;
};

#define CONCAT_(a, b) a##b
#define CONCAT(a, b) CONCAT_(a, b)

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


#define SCOPED_TIMER(var)                                        \
    ::tensorrt_custom_plugins::utils::ScopedTimer CONCAT(_timer, \
                                                        __LINE__)(var)

}  // namespace utils
}  // namespace tensorrt_custom_plugins

#endif

