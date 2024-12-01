cmake_minimum_required(VERSION 3.14)
project(gemm LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options for different implementations
option(WITH_MKL "Build with MKL implementation" OFF)
option(WITH_CUDA "Build with CUDA implementation" OFF)
option(WITH_OPENMP "Build with OpenMP implementation" OFF)
option(WITH_MPI "Build with MPI implementation" OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

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
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src/sequential)

if(WITH_MKL)
    add_subdirectory(src/mkl)
endif()

add_subdirectory(benchmark)

