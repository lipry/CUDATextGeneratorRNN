//
// Created by Fabio Lipreri on 2019-09-09.
//
#include <curand.h>
#include <curand_kernel.h>
#include "cublas_v2.h"
#include <cmath>
#include <vector>
#include "../utils/cudamath.h"
#include "RnnNetwork.h"
#include "../utils/common.h"
#include "RnnLayer.h"
#include "../operations/outputlayer.h"


RnnNetwork::RnnNetwork(int input_dim, int hidden_dim) : input_dim(input_dim), hidden_dim(hidden_dim)
{
    init_weights();
}

void RnnNetwork::init_weights() {
    U.allocate_size(hidden_dim, input_dim);
    W.allocate_size(hidden_dim, hidden_dim);
    V.allocate_size(input_dim, hidden_dim);
    //U.load_rand(-sqrt(1.0f / input_dim), sqrt(1.0f / input_dim));
    //W.load_rand(-sqrt(1.0f / hidden_dim), sqrt(1.0f / hidden_dim));
    //V.load_rand(-sqrt(1.0f / hidden_dim), sqrt(1.0f / hidden_dim));
    cout << "ENTRO IN INIT WEIGGTTISSSSS DI MERDA" << endl;
    U.load_value(0.4);
    W.load_value(0.5);
    V.load_value(-0.6);

    U.cpyHostToDev();
    W.cpyHostToDev();
    V.cpyHostToDev();
}
/*
 *     def forward_prop(self, x):
        cells = []
        h_prev = np.zeros(self.hidden_dim)
        for elem in x:
            v = OneHotEncodingUtilities.one_hot_encoder(elem, self.input_dim)
            cell = RNNLayer()
            cell.forward_pass(v, h_prev, self.U, self.W, self.V)
            h_prev = cell.h
            cells.append(cell)
        return cells
 */
vector<RnnLayer> RnnNetwork::forward_prop(cublasHandle_t handle, const std::vector<int> &x) {
    std::vector<RnnLayer> cells;
    //the allocation of shared pointer set values to 0.0 (I think)
    Matrix h_prev = Matrix(hidden_dim, 1);
    h_prev.allocate();
    //TODO: gestire meglio i trasferimenti in memoria
    //TODO: fare meglio l'init a zero (memset)
    for(int i=0;i<hidden_dim;i++){
        h_prev[i] = 0.0f;
    }
    h_prev.cpyHostToDev();

    Matrix v = Matrix(input_dim, 1);
    v.allocate();

    for (int elem : x){
        v.oneHotEncoder(elem);
        v.cpyHostToDev();
        RnnLayer cell = RnnLayer();
        cell.forward(handle, v, h_prev, this->U, this->W, this->V);
        h_prev = cell.getH();
        cells.push_back(cell);
    }

    /*for(std::vector<int>::size_type i = 0; i != cells.size(); i++) {
        cout << cells[i] << endl;
    }*/

    return cells;
}

/*
 *     def backprop_through_time(self, x, y, truncated=7):
        layers = self.forward_prop(x)
        T = len(layers)
        dU = np.zeros_like(self.U)
        dW = np.zeros_like(self.W)
        dV = np.zeros_like(self.V)

        output = OutputLayer()
        prev_ht = np.zeros(self.hidden_dim)
        diff_h = np.zeros(self.hidden_dim)
        for t in range(0, T):
            diff_Vprod = output.diff(layers[t].Vproduct, y[t])
            v = OneHotEncodingUtilities.one_hot_encoder(x[t], self.input_dim)
            _, dh_prev, dUt, dWt, dVt = layers[t].backward_pass(v, prev_ht, self.U, self.W, self.V, diff_h, diff_Vprod)
            prev_ht = layers[t].h
            diff_Vprod = np.zeros(self.input_dim)
            for i in range(t-1, max(t-1-truncated, -1), -1):
                v = OneHotEncodingUtilities.one_hot_encoder(x[i], self.input_dim)
                prev_hi = layers[i].h if i != 0 else np.zeros(self.hidden_dim)
                _, dh_prev, dUi, dWi, dVi = layers[i].backward_pass(v, prev_hi, self.U, self.W, self.V, dh_prev, diff_Vprod)
                dUt += dUi
                dWt += dWi
            dU += dUt
            dW += dWt
            dV += dVt
        return np.array([dU, dW, dV])
 */
void RnnNetwork::backprop_through_time(cublasHandle_t handle, const std::vector<int> &x, const std::vector<int> &y,
                                       int truncated) {
    std::vector<RnnLayer> layers = this->forward_prop(handle, x);
    //u.cpyDevToHost();
    //cout << u << endl;
    // porcodio
    Matrix dU = Matrix(hidden_dim, input_dim);
    dU.allocate();
    dU.init_with_zeroes();
    dU.cpyHostToDev();
    Matrix dW = Matrix(hidden_dim, hidden_dim);
    dW.allocate();
    dW.init_with_zeroes();
    dW.cpyHostToDev();
    Matrix dV = Matrix(input_dim, hidden_dim);
    dV.allocate();
    dV.init_with_zeroes();
    dV.cpyHostToDev();

    Matrix prev_ht = Matrix(hidden_dim, 1);
    prev_ht.allocate();
    prev_ht.init_with_zeroes();
    prev_ht.cpyHostToDev();

    Matrix prev_hi = Matrix(hidden_dim, 1);
    prev_hi.allocate();
    prev_hi.init_with_zeroes();
    prev_hi.cpyHostToDev();

    Matrix diff_h = Matrix(hidden_dim, 1);
    diff_h.allocate();
    diff_h.init_with_zeroes();
    diff_h.cpyHostToDev();

    Matrix v = Matrix(input_dim, 1);
    v.allocate();

    Matrix diff_Vprod = Matrix(input_dim, 1);
    diff_Vprod.allocate();

    OutputLayer output = OutputLayer();

    for(std::vector<int>::size_type t = 0; t < layers.size(); t++){
        //TODO: problema diff?
        diff_Vprod = output.diff(handle, layers[t].getOutput(), y[t]);
        v.oneHotEncoder(x[t]);
        v.cpyHostToDev();

        diff_Vprod.cpyDevToHost();
        cout << "diff_VProd" << endl;
        cout << diff_Vprod << endl;

        // TODO: NON FUNZIONA BACKWARD, risultati diversi da python
        layers[t].backward(handle, v, prev_ht, U, W, V, diff_h, diff_Vprod);
        Matrix dx = layers[t].getDx();
        Matrix dh_prev = layers[t].getDhPrev();
        Matrix dUt = layers[t].getDU();
        Matrix dWt = layers[t].getDW();
        Matrix dVt = layers[t].getDV();

        dx.cpyDevToHost();
        cout << "dx" << endl;
        cout << dx << endl;
        dh_prev.cpyDevToHost();
        cout << "dh_prev" << endl;
        cout << dh_prev << endl;
        dUt.cpyDevToHost();
        cout << "dUt" << endl;
        cout << dUt << endl;
        dWt.cpyDevToHost();
        cout << "dWt" << endl;
        cout << dWt << endl;
        dVt.cpyDevToHost();
        cout << "dVt" << endl;
        cout << dVt << endl;
        break;

        prev_ht = layers[t].getH();
        diff_Vprod.init_with_zeroes();


        for(int i = t-1; i < max(int(t)-1-truncated, -1); i--){
            v.oneHotEncoder(x[i]);
            v.cpyDevToHost();

            if (i!=0){
                prev_hi = layers[i].getH();
            }else{
                prev_hi.init_with_zeroes();
            }

            layers[i].backward(handle, v, prev_hi, this->U, this->W, this->V, dh_prev, diff_Vprod);
        }
    }




}

const Matrix &RnnNetwork::getU() const {
    return U;
}

const Matrix &RnnNetwork::getW() const {
    return W;
}

const Matrix &RnnNetwork::getV() const {
    return V;
}
