cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(CUDACXX /usr/local/cuda-8.0/bin/nvcc)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-8.0/bin/nvcc)

project(ProgettoGPU LANGUAGES CXX CUDA)

add_executable(ProgettoGPU main.cu src/linalgebra/linalgebra.cu src/operations/sigmoid.cu src/operations/sigmoid.h src/operations/matrix.cu src/operations/matrix.h src/utils/common.h)
target_link_libraries(ProgettoGPU -lcublas -lcurand)

target_compile_features(ProgettoGPU PUBLIC cxx_std_11)