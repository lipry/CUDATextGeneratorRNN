//
// Created by Fabio Lipreri on 2019-05-21.
//

#include "hyperbolic_tan.h"
#include "../utils/common.h"
#include <math.h>

__device__ float tanh_derivate(float x, float top_diff){
    return (1.0f - sqrt(x)) * top_diff;
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

Matrix& Tanh::forward(Matrix &v) {
    this->V = v;
    R.allocate_size(v.getX(), v.getY());

    dim3 TxB(BLOCK_SIZE);
    dim3 num_blocks((V.getY() * V.getX() + TxB.x - 1) / TxB.x);
    tanhForward<<<num_blocks, TxB>>>(R.getDevData().get(), V.getDevData().get(), R.getX(), R.getY());

    return R;
}

Matrix& Tanh::backward(Matrix &top_diff) {
    dX.allocate_size(R.getX(), R.getY());

    dim3 TxB(BLOCK_SIZE);
    dim3 num_blocks((R.getY() * R.getX() + TxB.x - 1) / TxB.x);
    tanhBackward<<<num_blocks, TxB>>>(dX.getDevData().get(), R.getDevData().get(),
            top_diff.getDevData().get(), R.getX(), R.getY());

    return dX;
}