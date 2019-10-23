//
// Created by Fabio Lipreri on 2019-09-09.
//

#ifndef PROGETTOGPU_RNNNETWORK_H
#define PROGETTOGPU_RNNNETWORK_H


#include "../utils/matrix.h"
#include "RnnLayer.h"
#include <vector>

class RnnNetwork {
private:
    int input_dim;
    int hidden_dim;
    Matrix U;
    Matrix W;
    Matrix V;

public:
    RnnNetwork(int input_dim, int hidden_dim);

    void init_weights();
    std::vector<RnnLayer> forward_prop(cublasHandle_t handle, const std::vector<int> &x);
    void backprop_through_time(cublasHandle_t handle, const std::vector<int> &x, const std::vector<int> &y,
                               int truncated);
    const Matrix &getU() const;
    const Matrix &getW() const;
    const Matrix &getV() const;
    const Matrix &getDu() const;
    const Matrix &getDw() const;
    const Matrix &getDv() const;

    const Matrix &getPrevHt() const;

    const Matrix &getDiffH() const;
};


#endif //PROGETTOGPU_RNNNETWORK_H
