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

void RnnLayer::forward(Matrix &x, Matrix &h_prev, Matrix &U, Matrix &W, Matrix &V){
    /*
*       self.Uproduct = mul.forward_pass(U, x)
        self.Wproduct = mul.forward_pass(M, h_prev)
        self.UWsum = add.forward_pass(self.Uproduct, self.Wproduct)
        self.h = tanh.forward_pass(self.UWsum)

        self.Vproduct = mul.forward_pass(V, self.h)
     */
    Matrix Uprod = Uproduct.forward(U, x);
    printf("Uprod: \n");
    Uprod.cpyDevToHost();
    Uprod.print_matrix();
    Matrix Wprod = Wproduct.forward(W, h_prev);
    printf("Wprod: \n");
    Wprod.cpyDevToHost();
    Wprod.print_matrix();
    Matrix UWs = UWsum.forward(Uprod, Wprod);
    printf("UWs: \n");
    UWs.cpyDevToHost();
    UWs.print_matrix();
    this->h = ht.forward(UWs);
    this->output = Vhproduct.forward(V, h);
}

void RnnLayer::backward(Matrix &x, Matrix &h_prev, Matrix &U, Matrix &W, Matrix &V, Matrix &diffh, Matrix &dVproduct){
    /*self.forward_pass(x, h_prev, U, M, V)
    dV, dhv = mul.backward_pass(V, self.h, dVproduct)
    dh = diffh + dhv
    dUWsum = tanh.backward_pass(self.UWsum, dh)
    dUWproduct, dWproduct = add.backward_pass(self.Uproduct, self.Wproduct, dUWsum)
    dU, dx = mul.backward_pass(U, x, dUWproduct)
    dM, dh_prev = mul.backward_pass(M, h_prev, dWproduct)
    return dx, dh_prev, dU, dM, dV*/

    this->forward(x, h_prev, U, W, V);
    Vhproduct.backward(dVproduct);
    Matrix dV = Vhproduct.getdMatrix();
    Matrix dhv = Vhproduct.getdVector();

    dV.cpyDevToHost();
    dhv.cpyDevToHost();

    printf("dV: \n");
    dV.print_matrix();
    printf("dhv: \n");
    dhv.print_matrix();

    /*Matrix dh;
    dh.allocate_size(dhv.getX(), dhv.getY());

    // dh = diffh + dhv
    dim3 TxB(BLOCK_SIZE);
    dim3 num_blocks((dhv.getY() * dhv.getX() + TxB.x - 1) / TxB.x);
    add_vect<<<num_blocks, TxB>>>(dh.getDevData().get(), dhv.getDevData().get(), diffh.getDevData().get(), dhv.getX(), diffh.getY());

    printf("dh: \n");
    dh.cpyDevToHost();
    dh.print_matrix();

    Matrix dUwsum = ht.backward(dh);

    printf("dUWsum: \n");
    dUwsum.cpyDevToHost();
    dUwsum.print_matrix();

    //dUWproduct == dWproduct nella derivata della somma
    Matrix dUWproduct = UWsum.backward(dUwsum);

    printf("------ dUWproduct: \n");
    dUWproduct.cpyDevToHost();
    dUWproduct.print_matrix();

    Wproduct.backward(dUWproduct);
    Matrix dh_prev = Wproduct.getdVector();
    Matrix dW = Wproduct.getdMatrix();

    printf("------ dh_prev: \n");
    dh_prev.cpyDevToHost();
    dh_prev.print_matrix();

    printf("------ dW: \n");
    dW.cpyDevToHost();
    dW.print_matrix();

    Uproduct.backward(dUWproduct);
    Matrix dx = Uproduct.getdVector();
    Matrix dU = Uproduct.getdMatrix();

    printf("------ dx: \n");
    dx.cpyDevToHost();
    dx.print_matrix();

    printf("------ dU: \n");
    dU.cpyDevToHost();
    dU.print_matrix();*/

}

const Matrix &RnnLayer::getH() const {
    return h;
}

const Matrix &RnnLayer::getOutput() const {
    return output;
}
