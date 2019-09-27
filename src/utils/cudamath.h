//
// Created by Fabio Lipreri on 2019-08-29.
//

#ifndef PROGETTOGPU_CUDAMATH_H
#define PROGETTOGPU_CUDAMATH_H

#include <curand.h>
#include <curand_kernel.h>

__global__ void add_vect(float *R, float *A, float *B, int x, int y);

__global__ void add_const_vect(float *R, float *A, float to_add, int x, int y);

__global__ void exp_predict(float *R, float *A, float max, int x, int y);

__global__ void outerProduct(float *Res, float *A, float *B, int N);

__device__ float tanh_derivate(float x, float top_diff);

__global__ void tanhForward(float* R, float* V, int x, int y);

__global__ void tanhBackward(float* dR, float* V, float *top_diff, int x, int y);

__device__ float sigmoid(float x);

__device__ float sigmoid_derivate(float x, float top_diff);

__global__ void sigmoidForward(float* R, float* V, int x, int y);

__global__ void sigmoidBackward(float* dR, float* V, float *top_diff, int x, int y);

__global__ void init_randoms(unsigned int seed, curandState_t* states);

__global__ void randoms(curandState_t* states, float* numbers, float lower, float higher);

#endif //PROGETTOGPU_CUDAMATH_H
