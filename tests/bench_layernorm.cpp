#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "layernorm/layernorm.h"
#include "utils/table_builder.h"
#include "utils/vector_check.h"

using namespace tensorrt_custom_plugins;

// ============================================================================
// Helper: benchmark a given GPU layernorm version and compare against CPU ref
// ============================================================================
layernorm::BenchmarkData bench_gpu_version(layernorm::Version version,
                                           const float *d_input,
                                           const float *d_gamma,
                                           const float *d_beta, float *d_output,
                                           float *d_mean, float *d_rstd,
                                           const std::vector<float> &h_output_ref,
                                           const std::vector<float> &h_mean_ref,
                                           const std::vector<float> &h_rstd_ref,
                                           int B, int T, int C, float eps,
                                           int block_size, float bytes,
                                           float flops, int avg_times = 2) {
    float avg_ms = 0.0f;
    for (int i = 0; i < avg_times; i++) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        layernorm::forward(version, d_input, d_gamma, d_beta, d_output, d_mean,
                           d_rstd, B, T, C, eps, nullptr, block_size);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        avg_ms += ms;
    }
    avg_ms /= avg_times;

    float bw_gb_s  = bytes / (avg_ms * 1e6f);
    float tflops   = flops / (avg_ms * 1e9f);

    // Copy device results back to host for error checking
    const int T_N = B * T;
    const int N   = B * T * C;
    std::vector<float> h_output(N);
    std::vector<float> h_mean(T_N);
    std::vector<float> h_rstd(T_N);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, N * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mean.data(), d_mean, T_N * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rstd.data(), d_rstd, T_N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float max_error       = utils::max_mse_vector_error<float>(h_output_ref, h_output);
    float max_error_mean  = utils::max_mse_vector_error<float>(h_mean_ref, h_mean);
    float max_error_rstd  = utils::max_mse_vector_error<float>(h_rstd_ref, h_rstd);

    return {/*model_config=*/"B=" + std::to_string(B) + ",T=" + std::to_string(T) +
                ",C=" + std::to_string(C),
            /*block_size=*/block_size,
            /*elements=*/B * T * C,
            /*time_ms=*/avg_ms,
            /*bandwidth_gbs=*/bw_gb_s,
            /*gflops=*/tflops,
            /*max_mse_error=*/std::max({max_error, max_error_mean, max_error_rstd})};
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::vector<int> block_sizes = {32, 96, 128, 256, 512, 1024};
    const float eps = 1e-5f;
    const int B = 2;
    const int T = 8192;
    const int C = 768;
    constexpr int T_N = B * T;
    constexpr int N = B * T * C;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

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

    float bytes = static_cast<float>((2 * N + 2 * C) * sizeof(float));
    float flops = 9.0f * static_cast<float>(N);
    TableBuilder table_builder;
    table_builder.columns({"Config", "Version", "Block Size", "Elements",
                           "Time(ms)", "BW(GB/s)", "TFLOPS", "Max MSE Error"});

    for (const int block_size : block_sizes) {
        for (int i = 0; i < N; ++i) {
            h_input[i] = dist(rng);
        }
        for (int i = 0; i < C; ++i) {
            h_gamma[i] = dist(rng);
            h_beta[i] = dist(rng);
        }

        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), C * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), C * sizeof(float),
                              cudaMemcpyHostToDevice));
        float cpu_bs_time;
        {
            SCOPED_TIMER(cpu_bs_time);
            layernorm::forward(layernorm::CPU, h_input.data(), h_gamma.data(),
                               h_beta.data(), h_output.data(), h_mean.data(),
                               h_rstd.data(), B, T, C, eps);
        }
        cpu_time += cpu_bs_time;

        for (auto version : {layernorm::NAIVE, layernorm::SHARED_MEMORY}) {
            auto result = bench_gpu_version(version, d_input, d_gamma, d_beta,
                                            d_output, d_mean, d_rstd,
                                            h_output, h_mean, h_rstd, B, T, C,
                                            eps, block_size, bytes, flops,
                                            /*avg_times=*/2);
            table_builder.row(
                result.model_config,
                version == layernorm::NAIVE ? "NAIVE" : "SHARED_MEMORY",
                result.block_size, result.elements, result.time_ms,
                result.bandwidth_gbs, result.gflops, result.max_mse_error);
        }
    }

    {
        cpu_time = cpu_time / block_sizes.size();
        float bw_gb_s = bytes / (cpu_time * 1e6f);  // GB/s
        float tflops = flops / (cpu_time * 1e9f);
        table_builder.row("B=" + std::to_string(B) + ",T=" + std::to_string(T) +
                              ",C=" + std::to_string(C),
                          "CPU", "-", B * T * C, cpu_time, bw_gb_s, tflops, "0");
    }

    table_builder.print();

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
