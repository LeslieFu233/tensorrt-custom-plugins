add_library(layernorm_plugin SHARED
    layernorm_kernel.cu
)

target_include_directories(layernorm_plugin PRIVATE
    # ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(layernorm_plugin PRIVATE
    tensorrt_custom_plugins_headers
    # ${NVINFER_LIB}
    # ${NVINFER_PLUGIN_LIB}
    ${CUDA_LIBRARIES}
    cudart
)

# Allow unresolved symbols at link time (TRT will resolve plugin symbols at load time)
if(APPLE)
    set_target_properties(layernorm_plugin PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()
