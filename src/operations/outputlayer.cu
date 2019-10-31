//
// Created by Fabio Lipreri on 2019-09-05.
//

/*
 * class OutputLayer:
    def predict(self, x):
        exp = np.exp(x-np.max(x))
        return exp / exp.sum(axis=0)

    def loss(self, x, y):
        p = self.predict(x)
        return -np.log(p[y])

    def diff(self, x, y):
        p = self.predict(x)
        p[y] -= 1
        return p
 */


#include <cmath>
#include "outputlayer.h"
#include "cublas_v2.h"
#include "../utils/common.h"
#include "../utils/cudamath.h"

void OutputLayer::predict(cublasHandle_t handle, const Matrix &x){
    float sum = 0.0f;
    float alpha;
    int maxindex = -1;

    if(!predictions.isDevAlloc()){
        predictions.allocate_size(x.getX(), x.getY());
    }

    CHECK_CUBLAS(cublasIsamax(handle, x.getX(), x.getDevData().get(), 1, &maxindex));
    dim3 TxB(BLOCK_SIZE);
    dim3 num_blocks((x.getY() * x.getX() + TxB.x - 1) / TxB.x);
    exp_predict<<<num_blocks, TxB>>>(predictions.getDevData().get(), x.getDevData().get(), x[maxindex-1], x.getX(), x.getY());
    CHECK_CUBLAS(cublasSasum(handle, predictions.getX(), predictions.getDevData().get(), 1, &sum))
    alpha = 1/sum;
    CHECK_CUBLAS(cublasSscal(handle, predictions.getX(), &alpha, predictions.getDevData().get(), 1))
}

float OutputLayer::loss(cublasHandle_t handle, const Matrix &x, int y) {
    if(!predictions.isDevAlloc()){
       this->predict(handle, x);
       predictions.cpyDevToHost();
    }
    //predictions.print_matrix();
    return -1.0 * log(predictions[y]);
}

const Matrix& OutputLayer::diff(cublasHandle_t handle, const Matrix &x, int y){
    if(!predictions.isDevAlloc()){
        this->predict(handle, x);
        predictions.cpyDevToHost();
    }
    predictions[y] -= 1;
    predictions.cpyHostToDev();
    return predictions;
}

const Matrix &OutputLayer::getPredictions() const {
    return predictions;
}
