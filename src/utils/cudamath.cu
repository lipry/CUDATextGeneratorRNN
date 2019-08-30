//
// Created by Fabio Lipreri on 2019-08-29.
//
#include "../utils/common.h"

__global__ void add_vect(float *R, float *A, float *B, int x, int y){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx < x*y)
        R[idx] = A[idx] + B[idx]; // TODO: fare un altra funzione __device__?
}

__global__ void outerProduct(float *Res, float *A, float *B, int N)
{
    int i, j, x, y;
    x = threadIdx.x;
    y = threadIdx.y;
    i = blockIdx.y*blockDim.y + y;
    j = blockIdx.x*blockDim.x + x;

    __shared__ float shrA[BLOCK_SIZE];
    __shared__ float shrB[BLOCK_SIZE];

    if (x == 0)
        shrA[y] = A[i];
    __syncthreads();

    if(y == 0)
        shrB[x] = B[j];
    __syncthreads();

    Res[i*N + j] = shrA[y]*shrB[x];

    //printf("----- outer [%d,%d],[%d,%d]: %f * %f = %f\n", x, y, i, j,shrA[y], shrB[x], Res[i*N + j]);
}

__device__ float tanh_derivate(float x, float top_diff){
    return (1.0f - x*x) * top_diff;
}

__global__ void tanhForward(float* R, float* V, int x, int y){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < x*y)
        R[index] = tanh(V[index]); //TODO: parallelizzare tanh
}

__global__ void tanhBackward(float* dR, float* V, float *top_diff, int x, int y){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < x*y)
        dR[index] = tanh_derivate(V[index], top_diff[index]);
}

__device__ float sigmoid(float x){
    return 1.0f / (1 + exp(-x)); //TODO: parallelizzare exp
}

__device__ float sigmoid_derivate(float x, float top_diff){
    return (1.0f - x)* x * top_diff;
}

__global__ void sigmoidForward(float* R, float* V, int x, int y){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < x*y)
        R[index] = sigmoid(V[index]);
}

__global__ void sigmoidBackward(float* dR, float* V, float *top_diff, int x, int y){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < x*y)
        dR[index] = sigmoid_derivate(V[index], top_diff[index]);
}