add_library(gemm_seq STATIC
    gemm_seq.cc
)

target_include_directories(gemm_seq
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

