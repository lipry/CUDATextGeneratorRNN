//
// Created by Fabio Lipreri on 2019-05-14.
//

#include <math.h>
#include "sigmoid.h"
#include "../utils/matrix.h"
#include "../utils/common.h"
#include <stdlib.h>
#include <stdio.h>

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

Matrix& Sigmoid::forward(Matrix &v){
    this->V = v;
    R.allocate_size(v.getX(), v.getY());

    dim3 TxB(BLOCK_SIZE);
    dim3 num_blocks((V.getY() * V.getX() + TxB.x - 1) / TxB.x);
    sigmoidForward<<<num_blocks, TxB>>>(R.getDevData(), V.getDevData(), R.getX(), R.getY());

    return R;

}

Matrix& Sigmoid::backward(Matrix &top_diff) {
    dX.allocate_size(R.getX(), R.getY());

    dim3 TxB(BLOCK_SIZE);
    dim3 num_blocks((R.getY() * R.getX() + TxB.x - 1) / TxB.x);
    sigmoidBackward<<<num_blocks, TxB>>>(dX.getDevData(), R.getDevData(),
            top_diff.getDevData(), R.getX(), R.getY());

    return dX;

}
