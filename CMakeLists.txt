cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(CUDACXX /usr/local/cuda-8.0/bin/nvcc)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-8.0/bin/nvcc)

project(ProgettoGPU LANGUAGES C CUDA)

add_executable(ProgettoGPU main.cu)