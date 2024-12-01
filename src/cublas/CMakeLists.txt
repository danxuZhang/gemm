find_package(CUDA REQUIRED)

cuda_add_library(gemm_cublas STATIC
    gemm_cublas.cu
)

target_include_directories(gemm_cublas
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(gemm_cublas
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
)
