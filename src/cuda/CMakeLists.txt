cuda_add_library(gemm_cuda STATIC
    gemm_cuda.cu
)

target_include_directories(gemm_cuda
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(gemm_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
