if(GPU_TARGETS MATCHES "gfx9")
    add_executable(client_grouped_gemm_fixed_nk_bias_fp16 grouped_gemm_fixed_nk_bias_fp16.cpp)
    target_link_libraries(client_grouped_gemm_fixed_nk_bias_fp16 PRIVATE composable_kernel::device_gemm_operations)
endif()
