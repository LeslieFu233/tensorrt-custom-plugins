# Correctness test
# add_executable(test_correctness test_correctness.cpp)
# target_include_directories(test_correctness PRIVATE
#     ${CUDA_INCLUDE_DIRS}
# )
# target_link_libraries(test_correctness
#     ${CUDA_LIBRARIES}
#     cudart
# )

# Benchmark
add_executable(bench_layernorm bench_layernorm.cpp)
target_include_directories(bench_layernorm PRIVATE
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(bench_layernorm PRIVATE
    tensorrt_custom_plugins_headers
    ${CUDA_LIBRARIES}
    CUDA::cudart
    layernorm_tensorrt_plugin
)
