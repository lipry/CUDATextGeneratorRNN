//
// Created by Fabio Lipreri on 2019-05-14.
//

#include <math.h>
#include "sigmoid.h"
#include "matrix.h"
#include "../utils/common.h"
#include <stdlib.h>
#include <stdio.h>

__device__ float sigmoid(float x){
    return 1.0f / (1 + exp(-x));
}

__global__ void sigmoidForward(float* R, float* V, int x, int y){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < x*y)
        R[index] = sigmoid(V[index]);
}

Matrix& Sigmoid::forward(Matrix &V){
    this->V = V;
    R.allocate_size(V.getX(), V.getY());

    dim3 TxB(BLOCK_SIZE);
    dim3 num_blocks((V.getY() * V.getX() + TxB.x - 1) / TxB.x);
    sigmoidForward<<<num_blocks, TxB>>>(R.getDevData().get(), V.getDevData().get(), R.getX(), R.getY());

    return R;

}
