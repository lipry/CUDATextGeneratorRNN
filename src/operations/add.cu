//
// Created by Fabio Lipreri on 2019-05-14.
//

#include "add.h"
#include "../utils/common.h"

__global__ void add_vect(float *R, float *A, float *B, int x, int y){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx < x*y)
        R[idx] = A[idx] + B[idx]; // TODO: fare un altra funzione __device__?
}

Matrix& Add::forward(Matrix& a, Matrix& b){
    this->A = a;
    this->B = b;
    R.allocate_size(a.getX(), a.getY());

    dim3 TxB(BLOCK_SIZE);
    dim3 num_blocks((A.getY() * A.getX() + TxB.x - 1) / TxB.x);
    add_vect<<<num_blocks, TxB>>>(R.getDevData(), A.getDevData(), B.getDevData(), A.getX(), A.getY());

    return R;
}

Matrix& Add::backward(Matrix& top_diff){
    this->dX = top_diff;
    return dX;
}