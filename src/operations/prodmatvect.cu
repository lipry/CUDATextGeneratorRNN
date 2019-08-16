//
// Created by Fabio Lipreri on 2019-05-31.
//

#include "cublas_v2.h"
#include "../utils/common.h"
#include "prodmatvect.h"

__global__ void outerProduct(float *Res, float *A, float *B, int N)
{
    int i, j, x, y;
    // Determine matrix element row i and column j.
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
}

Matrix& ProdMatVect::forward(Matrix& w, Matrix& v){
    /*if (w.getY() != v.getX())
        throw std::invalid_argument( "Matrix and Vectors dimension are not valid" );
    if(v.getY() != 1)
        throw std::invalid_argument( "V not a vector (Y != 1)" );*/

    this->W = w;
    this->V = v;
    R.allocate_size(w.getX(), 1);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    //printf("W.getX(): %d\n", W.getX());
    //printf("W.getY(): %d\n", W.getY());
    //printf("W: ");
    //W.print_matrix();
    //printf("V: ");
    //V.print_matrix();


    //W.print_matrix();

    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_T, W.getY(),
            W.getX(), &alpha, W.getDevData().get(), W.getY(),
            V.getDevData().get(), 1, &beta, R.getDevData().get(), 1));

    cublasDestroy(handle);

    return R;
}

void ProdMatVect::backward(Matrix &top_diff) {
    //TODO: CONTROLLARE TUTTO
    this->dW.allocate_size(W.getX(), W.getY());
    this->dv.allocate_size(top_diff.getX(), top_diff.getY());

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    //OK
    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_T, W.getY(),
                             W.getX(), &alpha, W.getDevData().get(), W.getY(),
                             top_diff.getDevData().get(), 1, &beta, dv.getDevData().get(), 1));
    cublasDestroy(handle);

    dim3 TxB(BLOCK_SIZE, BLOCK_SIZE);
    dim3 num_blocks(dW.getY()/TxB.x, dW.getX()/TxB.y);
    outerProduct<<<num_blocks, TxB>>>(dW.getDevData().get(), V.getDevData().get(),
            top_diff.getDevData().get(), W.getX());
}

Matrix& ProdMatVect::getdW() {
    return this->dW;
}

Matrix& ProdMatVect::getdv() {
    return this->dv;
}