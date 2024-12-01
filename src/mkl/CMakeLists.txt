add_library(gemm_mkl STATIC
    gemm_mkl.cc
)

target_include_directories(gemm_mkl
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(gemm_mkl
    PUBLIC ${MKL_LIBRARIES}
)

