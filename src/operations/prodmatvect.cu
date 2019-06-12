//
// Created by Fabio Lipreri on 2019-05-31.
//

#include "cublas_v2.h"
#include "../utils/common.h"
#include "prodmatvect.h"

__global__ void outerProduct(float *Res, float *A, float *B, int N)
{
    int i, j;
    // Determine matrix element row i and column j.
    i = blockIdx.y*blockDim.y + threadIdx.y;
    j = blockIdx.x*blockDim.x + threadIdx.x;

    // Each thread computes its own matrix element.
    printf("%f * %f = %f\n",A[i], B[j], A[i]*B[j]);
    Res[i*N + j] = A[i]*B[j];
}

/*__global__ void vect_vect(float *Res, float *A, float *B, int R, int C){
    int ROW = blockDim.y*blockIdx.y + threadIdx.y;
    int COL = blockDim.x*blockIdx.x + threadIdx.x;

    if((ROW < R) && (COL < C)){
        Res[ROW * C + COL] = 8;
        printf("(%d, %d, %d): %f* %f = %f \n", ROW, COL, ROW * C + COL);
    }else{
        printf("PISELLO (%d,%d) (%d,%d)\n", ROW, R, COL,C);//  idx = %f, idy = %f, x = %f, y = %f\n", idx, idy, x, y);
    }
}*/

Matrix& ProdMatVect::forward(Matrix& w, Matrix& v){
    /*if (w.getY() != v.getX())
        throw std::invalid_argument( "Matrix and Vectors dimension are not valid" );
    if(v.getY() != 1)
        throw std::invalid_argument( "V not a vector (Y != 1)" );*/

    this->W = w;
    this->V = v;
    R.allocate_size(v.getX(), 1);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, W.getX(),
            W.getY(), &alpha, W.getDevData().get(), W.getX(),
            V.getDevData().get(), 1, &beta, R.getDevData().get(), 1));

    cublasDestroy(handle);

    return R;
}

void ProdMatVect::backward(Matrix &top_diff) {
    this->dW.allocate_size(W.getX(), W.getY());
    this->dv.allocate_size(top_diff.getX(), top_diff.getY());

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));


    //OK
    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, W.getX(),
                             W.getY(), &alpha, W.getDevData().get(), W.getX(),
                             top_diff.getDevData().get(), 1, &beta, dv.getDevData().get(), 1));

    cublasDestroy(handle);

    dim3 TxB(2, 2);
    dim3 num_blocks(dW.getY()/TxB.x, dW.getX()/TxB.y);
    outerProduct<<<num_blocks, TxB>>>(dW.getDevData().get(), V.getDevData().get(),
            top_diff.getDevData().get(), W.getX());

    /*CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, V.getX(),
                             V.getY(), &alpha, V.getDevData().get(), V.getX(),
                            top_diff.getDevData().get(), 1, &beta, dW.getDevData().get(), 1));
    printf("V.getX = %d\n", V.getX());
    printf("V.getY = %d\n", V.getY());
    printf("top_diff.getX = %d\n", top_diff.getX());
    printf("top_diff.getY = %d\n", top_diff.getY());
    printf("V\n");
    V.print_matrix();
    printf("top_diff\n");
    top_diff.print_matrix();
    CHECK_CUBLAS(cublasSger(handle, dW.getX(), dW.getY(),
            &alpha, V.getDevData().get(), 1, top_diff.getDevData().get(), 1,
            dW.getDevData().get(), dW.getX()));

    cublasDestroy(handle);

    dim3 TxB(BLOCK_SIZE);
    dim3 num_blocks((top_diff.getY() * top_diff.getX() + TxB.x - 1) / TxB.x);
    elemwise_prod_vect<<<num_blocks, TxB>>>(dW.getDevData().get(), V.getDevData().get(),
                            top_diff.getDevData().get(), top_diff.getX(),
                            top_diff.getY());*/
}

Matrix& ProdMatVect::getdW() {
    return this->dW;
}

Matrix& ProdMatVect::getdv() {
    return this->dv;
}