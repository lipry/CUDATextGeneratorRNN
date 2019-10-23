//
// Created by Fabio Lipreri on 2019-06-14.
//
/*
 * class RNNLayer:
    def __init__(self):
        self.Uproduct = None
        self.Wproduct = None
        self.UWsum = None
        self.h = None
        self.Vproduct = None

    def forward_pass(self, x, h_prev, U, M, V):
        # calculating the state function h
        self.Uproduct = mul.forward_pass(U, x)
        self.Wproduct = mul.forward_pass(M, h_prev)
        self.UWsum = add.forward_pass(self.Uproduct, self.Wproduct)
        self.h = tanh.forward_pass(self.UWsum)

        self.Vproduct = mul.forward_pass(V, self.h)

    def backward_pass(self, x, h_prev, U, M, V, diffh, dVproduct):
        self.forward_pass(x, h_prev, U, M, V)
        dV, dhv = mul.backward_pass(V, self.h, dVproduct)
        dh = diffh + dhv
        dUWsum = tanh.backward_pass(self.UWsum, dh)
        dUproduct, dWproduct = add.backward_pass(self.Uproduct, self.Wproduct, dUWsum)
        dU, dx = mul.backward_pass(U, x, dUproduct)
        dM, dh_prev = mul.backward_pass(M, h_prev, dWproduct)
        return dx, dh_prev, dU, dM, dV
 */


#include "RnnLayer.h"
#include "../utils/common.h"
#include "../utils/cudamath.h"


void RnnLayer::forward(cublasHandle_t handle, Matrix &x, Matrix &h_prev, Matrix &U, Matrix &W, Matrix &V){
    Matrix Uprod = Uproduct.forward(handle, U, x);
    Matrix Wprod = Wproduct.forward(handle, W, h_prev);
    Matrix UWs = UWsum.forward(Uprod, Wprod);
    this->h = ht.forward(UWs);
    this->output = Vhproduct.forward(handle, V, h);
}

void RnnLayer::backward(cublasHandle_t handle, Matrix &x, Matrix &h_prev, Matrix &U,
        Matrix &W, Matrix &V, Matrix &diffh, Matrix &dVproduct){

    this->forward(handle, x, h_prev, U, W, V);
    Vhproduct.backward(handle, dVproduct);
    this->dV = Vhproduct.getdMatrix();
    Matrix dhv = Vhproduct.getdVector();

    Matrix dh;
    dh.allocate_size(dhv.getX(), dhv.getY());

    dim3 TxB(BLOCK_SIZE);
    dim3 num_blocks((dhv.getY() * dhv.getX() + TxB.x - 1) / TxB.x);
    add_vect<<<num_blocks, TxB>>>(dh.getDevData().get(), dhv.getDevData().get(),
            diffh.getDevData().get(), dhv.getX(), diffh.getY());

    Matrix dUwsum = ht.backward(dh);
    Matrix dUWproduct = UWsum.backward(dUwsum);
    Uproduct.backward(handle, dUWproduct);


    this->dx = Uproduct.getdVector();
    this->dU = Uproduct.getdMatrix();
    Wproduct.backward(handle, dUWproduct);
    this->dh_prev = Wproduct.getdVector();
    this->dW = Wproduct.getdMatrix();
}

const Matrix &RnnLayer::getH() const {
    return h;
}

const Matrix &RnnLayer::getOutput() const {
    return output;
}

const Matrix &RnnLayer::getDx() const {
    return dx;
}

const Matrix &RnnLayer::getDhPrev() const {
    return dh_prev;
}

const Matrix &RnnLayer::getDU() const {
    return dU;
}

const Matrix &RnnLayer::getDW() const {
    return dW;
}

const Matrix &RnnLayer::getDV() const {
    return dV;
}

const ProdMatVect &RnnLayer::getUproduct() const {
    return Uproduct;
}

const ProdMatVect &RnnLayer::getWproduct() const {
    return Wproduct;
}

const Add &RnnLayer::getUWsum() const {
    return UWsum;
}

const Tanh &RnnLayer::getHt() const {
    return ht;
}

const ProdMatVect &RnnLayer::getVhproduct() const {
    return Vhproduct;
}

std::ostream& operator<<(std::ostream &strm, const RnnLayer &cell) {
    cell.getH().cpyDevToHost();
    strm << "h: " << cell.getH() << endl;
    cell.getOutput().cpyDevToHost();
    strm << "output: " << cell.getOutput() << endl;
    cell.getDx().cpyDevToHost();
    strm << "dx: " << cell.getDx()<< endl;
    cell.getDhPrev().cpyDevToHost();
    strm << "dh_prev: " << cell.getDhPrev() << endl;
    cell.getDU().cpyDevToHost();
    strm << "dU: " << cell.getDU() << endl;
    cell.getDW().cpyDevToHost();
    strm << "dW: " << cell.getDW() << endl;
    cell.getDV().cpyDevToHost();
    strm << "dV: " << cell.getDV() << endl;
    return strm;
}