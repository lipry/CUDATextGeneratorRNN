cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

set(CUDACXX /usr/local/cuda-8.0/bin/nvcc)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-8.0/bin/nvcc)

project(ProgettoGPU LANGUAGES C CXX CUDA)

add_executable(ProgettoGPU main.cu src/linalgebra/linalgebra.cu src/operations/sigmoid.cu src/operations/sigmoid.h src/utils/matrix.cu src/utils/matrix.h src/utils/common.h src/operations/add.cu src/operations/add.h src/operations/hyperbolic_tan.cpp src/operations/hyperbolic_tan.h)

target_compile_features(ProgettoGPU PUBLIC cxx_std_14)