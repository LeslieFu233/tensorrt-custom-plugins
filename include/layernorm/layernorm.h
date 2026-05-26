#ifndef TENSORRT_CUSTOM_PLUGINS_LAYERNORM_H
#define TENSORRT_CUSTOM_PLUGINS_LAYERNORM_H

#include <cuda_runtime.h>
#include "utils/utils.h"
namespace tensorrt_custom_plugins{
namespace layernorm{

enum Version
{
    CPU = 0,
    NAIVE,
};

/// @brief 
/// @param version 
/// @param input 
/// @param weight 
/// @param bias 
/// @param output 
/// @param B input batch size
/// @param T input sample size
/// @param C input sample's feature dimension
/// @param eps learning rate
/// @param stream cuda stream
void forward
(
    Version version,
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    float* mean,
    float* rstd,
    int B,
    int T,
    int C,
    float eps,
    cudaStream_t stream=nullptr,
    int block_size=128
);

struct BenchmarkData {
    std::string model_config;
    int block_size;
    int elements;
    double time_ms;
    double bandwidth_gbs;
    double gflops;
    double max_mse_error;
};

} // namespace layernorm
} // namespace tensorrt_custom_plugins

#endif