#include "layernorm/layernorm.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

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
                              float* rstd, int B, int T, int C, float eps) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int N = B * T;
    if (idx >= N) return;  // guard against partial last block

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

__global__ void mean_shared_knl(const float * input, float* mean, int C) {
    extern __shared__ float mem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x;
    int block_size = blockDim.x;

    const float* x_block = input + idx * C;
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sum += x_block[i];
    }
    mem[tid] = sum;
    __syncthreads();
    for (int stride = block_size >> 1; stride >= 1; stride = stride >> 1) {
        __syncthreads();
        if (tid < stride) mem[tid] += mem[tid + stride];
    }
    if (tid == 0) {
        mean[idx] = mem[0] / C;
    }
}
__global__ void rstd_shared_knl(const float* input, const float* mean, float * rstd, int C) {
    extern __shared__ float shared[];
    int idx = blockIdx.x; // range [0, B*T)
    int tid = threadIdx.x; // range [0, block_size)
    int block_size = blockDim.x;
    const float* x = input + idx * C;
    float m = mean[idx];
    // thread coarsening
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    shared[tid] = sum;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        rstd[idx] = 1.0f / sqrtf(shared[0] / C + 1e-5f);
    }
}

__global__ void forward_shared_mem(const float* input, const float* weight,
                              const float* bias, float* output, const float* mean,
                              const float* rstd, int B, int T, int C, float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;
    if (idx >= N) return;  // guard against partial last block

    int bt = idx / C;
    int c = idx % C;

    float m = mean[bt];
    float s = rstd[bt];
    float xi = input[idx];
    float n = s * (xi - m);
    float o = n * weight[c] + bias[c];

    output[idx] = o;
}

__global__ void forward_coorperative_groups(const float* input, const float* weight,
                              const float* bias, float* output, float* mean,
                              float* rstd, int B, int T, int C, float eps)
{
    namespace cp = cooperative_groups;
}
__host__ void forward(Version version, const float* input, const float* weight,
                      const float* bias, float* output, float* mean,
                      float* rstd, int B, int T, int C, float eps,
                      cudaStream_t stream, int block_size) {
    switch (version) {
        case CPU: {
            forward_cpu(input, weight, bias, output, mean, rstd, B, T, C, eps);
            break;
        }
        case NAIVE: {
            const int grid_size = CEIL_DIV(B * T, block_size);
            if (stream != nullptr)
                forward_naive<<<grid_size, block_size, 0, stream>>>(
                    input, weight, bias, output, mean, rstd, B, T, C, eps);
            else
                forward_naive<<<grid_size, block_size>>>(
                    input, weight, bias, output, mean, rstd, B, T, C, eps);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            break;
        }
        case SHARED_MEMORY: {
            mean_shared_knl<<<B * T, block_size, block_size * sizeof(float)>>>(input, mean, C);
            CUDA_CHECK(cudaGetLastError());
            rstd_shared_knl<<<B * T, block_size, block_size * sizeof(float)>>>(input, mean, rstd, C);
            CUDA_CHECK(cudaGetLastError());
            const int block_size2 = 256;
            const int grid_size = CEIL_DIV(B * T * C, block_size2);
            forward_shared_mem<<<grid_size, block_size2>>>(input, weight, bias, output, mean, rstd, B, T, C, eps);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        default: {
            std::cerr << "Unsupported version!" << std::endl;
            std::abort();
            break;
        }
    }
}

}  // namespace layernorm
}  // namespace tensorrt_custom_plugins
