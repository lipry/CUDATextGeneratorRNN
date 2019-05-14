//
// Created by Fabio Lipreri on 2019-05-14.
//

#include <math.h>
#include "sigmoid.h"
#include <stdlib.h>
#include <stdio.h>

__device__ float sigmoid(float x){
    return 1.0f / (1 + exp(-x));
}

__global__ void sigmoidForward(float* R, float* V, int xdim){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < xdim)
        R[index] = sigmoid(V[index]);
}

void Sigmoid::forward(){
    int N = 12;
    float* A;
    float* B;
    float* d_A;
    float* d_B;
    int nBytes = N*sizeof(float);

    A = (float*)malloc(nBytes);
    B = (float*)malloc(nBytes);

    cudaMalloc((void **)&d_A, nBytes);
    cudaMalloc((void **)&d_B, nBytes);

    for(int i = 0; i < N; i++){
        B[i] = (float) i +1;
    }

    for(int i = 0; i < N; i++){
        printf("%f ", B[i]);
    }
    printf("\n");

    cudaMemcpy(d_B, B, nBytes, cudaMemcpyHostToDevice);

    sigmoidForward<<<N/3, 3>>>(d_A, d_B, N);

    cudaMemcpy(A, d_A, nBytes, cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++){
        printf("%f ", A[i]);
    }

    free(A);
    free(B);

    cudaFree(d_A);
    cudaFree(d_B);

}
