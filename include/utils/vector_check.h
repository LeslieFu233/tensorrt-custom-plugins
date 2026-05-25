#ifndef TENSORRT_CUSTOM_PLUGIN_VECTOR_CHECK_H
#define TENSORRT_CUSTOM_PLUGIN_VECTOR_CHECK_H

#include <vector>
#include <cmath>
namespace tensorrt_custom_plugins {
namespace utils {
template <typename T>
bool is_vector_close(const std::vector<T>& a, const std::vector<T>& b,
                     float rtol = 1e-5) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (std::fabs(a[i] - b[i]) >
            rtol * std::max(std::fabs(a[i]), std::fabs(b[i])))
            return false;
    }
    return true;
}

template <typename T>
T max_mse_vector_error(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return false;
    float max_error = -0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float error = std::fabs(a[i] - b[i])*std::fabs(a[i] - b[i]);
        max_error = std::max(error, max_error);
    }
    return static_cast<T>(max_error);
}
}  // namespace utils
}  // namespace tensorrt_custom_plugins
#endif