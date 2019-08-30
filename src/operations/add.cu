//
// Created by Fabio Lipreri on 2019-05-14.
//


#include "add.h"
#include "cublas_v2.h"
#include "../utils/common.h"
#include "../utils/cudamath.h"

Matrix& Add::forward(Matrix& a, Matrix& b){
    this->A = a;
    this->B = b;
    R.allocate_size(a.getX(), a.getY());

    dim3 TxB(BLOCK_SIZE);
    dim3 num_blocks((A.getY() * A.getX() + TxB.x - 1) / TxB.x);
    add_vect<<<num_blocks, TxB>>>(R.getDevData().get(), A.getDevData().get(), B.getDevData().get(), A.getX(), A.getY());

    return R;
}

Matrix& Add::backward(Matrix& top_diff){
    this->dA = top_diff;
    return dA;
}