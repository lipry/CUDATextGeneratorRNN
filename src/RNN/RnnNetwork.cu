//
// Created by Fabio Lipreri on 2019-09-09.
//
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>
#include "../utils/cudamath.h"
#include "RnnNetwork.h"
#include "../utils/common.h"


RnnNetwork::RnnNetwork(int input_dim, int hidden_dim) : input_dim(input_dim), hidden_dim(hidden_dim)
{
    U.allocate_size(hidden_dim, input_dim);
    W.allocate_size(hidden_dim, hidden_dim);
    V.allocate_size(input_dim, hidden_dim);
    init_weights();
}

void RnnNetwork::init_weights() {
    U.load_rand(-sqrt(1.0f / input_dim), sqrt(1.0f / input_dim));
    W.load_rand(-sqrt(1.0f / hidden_dim), sqrt(1.0f / hidden_dim));
    V.load_rand(-sqrt(1.0f / hidden_dim), sqrt(1.0f / hidden_dim));
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
