cmake_minimum_required(VERSION 3.14)
project(gemm LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options for different implementations
option(WITH_MKL "Build with MKL implementation" OFF)
option(WITH_CUDA "Build with CUDA implementation" OFF)
option(WITH_OPENMP "Build with OpenMP implementation" OFF)
option(WITH_MPI "Build with MPI implementation" OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math -funroll-loops")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    # For Intel compiler
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 ")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 ")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fp-model fast=2 -unroll")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
endif()

# Find required packages
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
