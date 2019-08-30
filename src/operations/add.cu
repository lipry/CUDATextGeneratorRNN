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
//    Matrix a_like;
//    Matrix b_like;
//    a_like.matrix_like(1.0f, A);
//    b_like.matrix_like(1.0f, B);
//    a_like.print_matrix();
//    b_like.print_matrix();
//    a_like.cpyHostToDev();
//    b_like.cpyHostToDev();
//
//    this->dA.allocate_size(a_like.getX(), 1);
//    this->dB.allocate_size(b_like.getX(), 1);
//
//    float alpha = 1.0f;
//    float beta = 0.0f;
//
//    cublasHandle_t handle;
//    CHECK_CUBLAS(cublasCreate(&handle));
//
//    size_t m = a_like.getY();
//    size_t n = a_like.getX();
//    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, m, n,
//                             &alpha, a_like.getDevData().get(), m, top_diff.getDevData().get(), 1,
//                             &beta, dA.getDevData().get(), 1));
//
//    cublasDestroy(handle);

    return dA;
}