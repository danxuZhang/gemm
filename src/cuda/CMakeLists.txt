enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Create CUDA library
add_library(gemm_cuda STATIC
    gemm_cuda.cu
)

target_include_directories(gemm_cuda
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(gemm_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
