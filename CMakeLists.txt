cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

set(CUDACXX /usr/local/cuda-8.0/bin/nvcc)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-8.0/bin/nvcc)

project(ProgettoGPU LANGUAGES C CXX CUDA)

add_executable(ProgettoGPU main.cu src/linalgebra/linalgebra.cu src/operations/sigmoid.cu src/operations/sigmoid.h src/utils/matrix.cu src/utils/matrix.h src/utils/common.h src/operations/add.cu src/operations/add.h src/operations/hyperbolic_tan.cu src/operations/hyperbolic_tan.h dummy.cpp src/utils/common.cu src/operations/prodmatvect.cu src/operations/prodmatvect.h src/RNN/RnnLayer.cu src/RNN/RnnLayer.h src/utils/cudamath.cu src/utils/cudamath.h src/operations/outputlayer.cu src/operations/outputlayer.h src/RNN/RnnNetwork.cu src/RNN/RnnNetwork.h)

target_compile_features(ProgettoGPU PUBLIC cxx_std_14)
target_link_libraries(ProgettoGPU -lcublas -lcurand)