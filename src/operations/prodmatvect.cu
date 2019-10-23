//
// Created by Fabio Lipreri on 2019-05-31.
//

#include "cublas_v2.h"
#include "../utils/common.h"
#include "prodmatvect.h"
#include "../utils/cudamath.h"


Matrix& ProdMatVect::forward(cublasHandle_t handle, Matrix& w, Matrix& v){
    /*if (w.getY() != v.getX())
        throw std::invalid_argument( "Matrix and Vectors dimension are not valid" );
    if(v.getY() != 1)
        throw std::invalid_argument( "V not a vector (Y != 1)" );*/

    this->M = w;
    this->V = v;
    R.allocate_size(w.getX(), 1);

    float alpha = 1.0f;
    float beta = 0.0f;

    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_T, M.getY(),
            M.getX(), &alpha, M.getDevData().get(), M.getY(),
            V.getDevData().get(), 1, &beta, R.getDevData().get(), 1));

    return R;
}

void ProdMatVect::backward(cublasHandle_t handle, Matrix &top_diff) {
    this->dM.allocate_size(top_diff.getX(), V.getX());
    this->dv.allocate_size(M.getY(), top_diff.getY());

    float alpha = 1.0f;
    float beta = 0.0f;
    //dv
    size_t m = M.getY();
    size_t n = M.getX();
    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, m, n,
            &alpha, M.getDevData().get(), m, top_diff.getDevData().get(), 1, &beta, dv.getDevData().get(), 1));

    //dM
    dim3 TxB(BLOCK_SIZE, BLOCK_SIZE);
    dim3 num_blocks(ceil(float(dM.getY())/TxB.x), ceil(float(dM.getX())/TxB.y));
    outerProduct<<<num_blocks, TxB>>>(dM.getDevData().get(), top_diff.getDevData().get(),
            V.getDevData().get(), dM.getY());
}

Matrix& ProdMatVect::getdMatrix() {
    return this->dM;
}

Matrix& ProdMatVect::getdVector() {
    return this->dv;
}