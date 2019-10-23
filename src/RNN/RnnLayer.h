//
// Created by Fabio Lipreri on 2019-06-14.
//

#ifndef PROGETTOGPU_RNNLAYER_H
#define PROGETTOGPU_RNNLAYER_H

#include "cublas_v2.h"
#include "../operations/prodmatvect.h"
#include "../operations/add.h"
#include "../operations/hyperbolic_tan.h"

class RnnLayer {
private:
    ProdMatVect Uproduct;
    ProdMatVect Wproduct;
    Add UWsum;
    Tanh ht;
    ProdMatVect Vhproduct;
    Matrix h;
    Matrix output;
    Matrix dx;
    Matrix dh_prev;
    Matrix dU;
    Matrix dW;
    Matrix dV;
public:
    void forward(cublasHandle_t handle, Matrix &x, Matrix &h_prev, Matrix &U, Matrix &W, Matrix &V);
    void backward(cublasHandle_t handle, Matrix &x, Matrix &h_prev, Matrix &U, Matrix &W, Matrix &V, Matrix &diffh, Matrix &dVproduct);

    const ProdMatVect &getUproduct() const;

    const ProdMatVect &getWproduct() const;

    const Add &getUWsum() const;

    const Tanh &getHt() const;

    const ProdMatVect &getVhproduct() const;

    const Matrix &getH() const;
    const Matrix &getOutput() const;
    const Matrix &getDx() const;
    const Matrix &getDhPrev() const;
    const Matrix &getDU() const;
    const Matrix &getDW() const;
    const Matrix &getDV() const;
};

std::ostream& operator<<(std::ostream &strm,const RnnLayer &cell);


#endif //PROGETTOGPU_RNNLAYER_H
