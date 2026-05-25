#include "layernorm/layernorm.h"

namespace tensorrt_custom_plugins {
namespace layernorm {
__host__ void forward_cpu(const float* input, const float* weight,
                          const float* bias, float* output, float* mean,
                          float* rstd, int B, int T, int C, float eps) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // 按照样本迭代，每一个样本按特征维度独立归一化，这是代表了特征向量的数组
            const float* x = input + b * T * C + t * C;
            float m = 0.0f;
            for (int c = 0; c < C; c++) {
                m += x[c];
            }
            m = m / (float)C;

            float sigma = 0.0f;
            for (int c = 0; c < C; c++) {
                float diff = x[c] - m;
                sigma += diff * diff;
            }
            sigma = sigma / (float)C;

            float s = 1.0f / sqrtf(sigma + eps);
            float* out_begin = output + b * T * C + t * C;
            for (int c = 0; c < C; c++) {
                float diff = x[c] - m;
                out_begin[c] = diff * s * weight[c] + bias[c];
            }
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

__global__ void forward_naive(const float* input, const float* weight,
                              const float* bias, float* output, float* mean,
                              float* rstd, int C, float eps) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const float* input_init = input + idx * C;
    float m = 0.0f;
    for (int c = 0; c < C; c++) {
        m += input_init[c];
    }
    m = m / (float)C;

    float sigma = 0.0f;
    for (int c = 0; c < C; c++) {
        float diff = input_init[c] - m;
        sigma += diff * diff;
    }
    sigma = sigma / (float)C;

    float s = 1.0f / sqrtf(sigma + eps);
    float* out_begin = output + idx * C;
    for (int c = 0; c < C; c++) {
        float diff = input_init[c] - m;
        out_begin[c] = diff * s * weight[c] + bias[c];
    }
    mean[idx] = m;
    rstd[idx] = s;
}

__host__ void forward(Version version, const float* input, const float* weight,
                      const float* bias, float* output, float* mean,
                      float* rstd, int B, int T, int C, float eps,
                      cudaStream_t stream, int block_size) {
    switch (version) {
        case CPU:
        {
            forward_cpu(input, weight, bias, output, mean, rstd, B, T, C, eps);
            break;
        }
        case NAIVE:
        {
            const int grid_size = CEIL_DIV(B * T, block_size);
            if(stream!=nullptr)
                forward_naive<<<grid_size, block_size, 0, stream>>>(
                input, weight, bias, output, mean, rstd, C, eps);
            else
                forward_naive<<<grid_size, block_size>>>(
                input, weight, bias, output, mean, rstd, C, eps);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            break;
        }
        default:
        {
            std::cerr << "Unsupported version!" << std::endl;
            std::abort();
            break;
        }
            
    }
}

}  // namespace layernorm
}  // namespace tensorrt_custom_plugins
