add_executable(benchmark_gemm
    main.cc
)

target_link_libraries(benchmark_gemm
    PRIVATE gemm_seq
)

if(WITH_MKL)
    target_compile_definitions(benchmark_gemm PRIVATE WITH_MKL)
    target_link_libraries(benchmark_gemm
        PRIVATE gemm_mkl
    )
endif()

