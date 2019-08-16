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

    def forward_pass(self, x, h_prev, U, W, V):
        # calculating the state function h
        self.Uproduct = mul.forward_pass(U, x)
        self.Wproduct = mul.forward_pass(W, h_prev)
        self.UWsum = add.forward_pass(self.Uproduct, self.Wproduct)
        self.h = tanh.forward_pass(self.UWsum)

        self.Vproduct = mul.forward_pass(V, self.h)

    def backward_pass(self, x, h_prev, U, W, V, diffh, dVproduct):
        self.forward_pass(x, h_prev, U, W, V)
        dV, dhv = mul.backward_pass(V, self.h, dVproduct)
        dh = diffh + dhv
        dUWsum = tanh.backward_pass(self.UWsum, dh)
        dUproduct, dWproduct = add.backward_pass(self.Uproduct, self.Wproduct, dUWsum)
        dU, dx = mul.backward_pass(U, x, dUproduct)
        dW, dh_prev = mul.backward_pass(W, h_prev, dWproduct)
        return dx, dh_prev, dU, dW, dV
 */


#include "RnnLayer.h"

void RnnLayer::forward(Matrix &x, Matrix &h_prev, Matrix &U, Matrix &W, Matrix &V){
    /*
*       self.Uproduct = mul.forward_pass(U, x)
        self.Wproduct = mul.forward_pass(W, h_prev)
        self.UWsum = add.forward_pass(self.Uproduct, self.Wproduct)
        self.h = tanh.forward_pass(self.UWsum)

        self.Vproduct = mul.forward_pass(V, self.h)
     */
    Matrix Uprod = Uproduct.forward(U, x);
    Matrix Wprod = Wproduct.forward(W, h_prev);
    Matrix UWs = UWsum.forward(Uprod, Wprod);
    Matrix h = ht.forward(UWs);

    Matrix Vprod = Vhproduct.forward(V, h);

    Uprod.cpyDevToHost();
    Wprod.cpyDevToHost();
    UWs.cpyDevToHost();
    h.cpyDevToHost();
    Vprod.cpyDevToHost();
}