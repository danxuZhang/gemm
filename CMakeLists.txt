cmake_minimum_required(VERSION 3.18)
project(gemm LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -march=native")

option(WITH_SEQ "Build with sequential implementation" ON)
option(WITH_CUDA "Build with CUDA implementation" OFF)
option(WITH_OPENMP "Build with OpenMP implementation" OFF)
option(WITH_MPI "Build with MPI implementation" OFF)
option(WITH_MKL "Build with MKL implementation" OFF)
option(WITH_CUBLAS "Build with CuBLAS implementation" OFF)


include_directories(${PROJECT_SOURCE_DIR}/include)

if (WITH_SEQ)
    add_subdirectory(src/sequential)
endif()

if(WITH_MKL)
    if(NOT DEFINED ENV{MKLROOT})
        message(FATAL_ERROR "MKLROOT environment variable is not set. Please load MKL module first.")
    endif()
    
    include_directories($ENV{MKLROOT}/include)
    link_directories($ENV{MKLROOT}/lib/intel64)
    
    set(MKL_LIBRARIES 
        -Wl,--start-group
        -lmkl_intel_lp64 
        -lmkl_sequential 
        -lmkl_core
        -Wl,--end-group
        -lpthread 
        -lm
        -ldl
    )
    add_subdirectory(src/mkl)
endif()

if(WITH_CUDA OR WITH_CUBLAS)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CUDA_ARCHITECTURES "sm90")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
endif()

if(WITH_CUBLAS)
    add_subdirectory(src/cublas)
endif()

if(WITH_CUDA)
    add_subdirectory(src/cuda)
endif()

add_subdirectory(benchmark)
