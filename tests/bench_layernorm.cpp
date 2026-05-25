#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <chrono>
#include "layernorm/layernorm.h"
#include "utils/vector_check.h"
using namespace tensorrt_custom_plugins;

// ============================================================================
// Main
// ============================================================================
int main() {
    std::vector<int> block_sizes = {32,64,128,256,512,1024};
    const int avg_times = 20;
    const float eps = 1e-5f;
    const int B = 64;
    const int T = 10000;
    const int C = 768;
    constexpr int T_N = B * T;
    constexpr int N = B * T * C;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::printf("%-24s %8s %10s %10s %10s %6s\n", "Shape", "Elements", "Time(ms)",
                "BW(GB/s)", "TFlops", "Max MSE Error");
    std::printf("%s\n", std::string(80, '-').c_str());
    
    std::vector<float> h_input(N);
    std::vector<float> h_gamma(C);
    std::vector<float> h_beta(C);
    std::vector<float> h_output(N);
    std::vector<float> h_mean(T_N);
    std::vector<float> h_rstd(T_N);
    float *d_input, *d_gamma, *d_beta, *d_output, *d_mean, *d_rstd;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, T_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rstd, T_N * sizeof(float)));

    float cpu_time = 0.0f;
    for (const int block_size : block_sizes) {
        for (int i = 0; i < N; ++i) {
            h_input[i] = dist(rng);
        }
        for (int i = 0; i < C; ++i) {
            h_gamma[i] = dist(rng);
            h_beta[i] = dist(rng);
        }
        std::printf("block_size = %d\n", block_size);

        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), C * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), C * sizeof(float),
                              cudaMemcpyHostToDevice));
        layernorm::forward(layernorm::CPU, h_input.data(), h_gamma.data(),
                           h_beta.data(), h_output.data(), h_mean.data(),
                           h_rstd.data(), B, T, C, eps);
        float avg_ms = 0.0f;
        for(int i=0;i<20;i++)
        {
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            CUDA_CHECK(cudaEventRecord(start));
            layernorm::forward(layernorm::NAIVE, d_input, d_gamma, d_beta, d_output,
                            d_mean, d_rstd, B, T, C, eps);

            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            avg_ms += ms;
        }
        // Timing
        avg_ms = avg_ms/20;
        // Bandwidth: read input + read gamma + read beta + write output = (N +
        // C + C + N) * 4 bytes
        float bytes = static_cast<float>((2 * N + 2 * C) * sizeof(float));
        float bw_gb_s = bytes / (avg_ms * 1e6f);  // GB/s

        // FLOPS: mean (C adds), var (C subs + C muls + C adds), norm (C subs +
        // C muls + C fmas) ~ 9 * C per row => 9 * N total
        float flops = 9.0f * static_cast<float>(N);
        float tflops = flops / (avg_ms * 1e9f);
        
        std::vector<float> h_output_copy(N);
        std::vector<float> h_mean_copy(T_N);
        std::vector<float> h_rstd_copy(T_N);
        CUDA_CHECK(cudaMemcpy(h_output_copy.data(), d_output, N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mean_copy.data(), d_mean, T_N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mean_copy.data(), d_rstd, T_N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        float max_error =
            utils::max_mse_vector_error<float>(h_output, h_output_copy);
        char label[32];
        std::snprintf(label, sizeof(label), "B=%d T=%d, C=%d T", B, T, C);
        std::printf("%-24s %8d %10.4f %10.2f %10.4f %6.4f\n", label, N, avg_ms, bw_gb_s,
                    tflops, max_error);
        
    }
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
