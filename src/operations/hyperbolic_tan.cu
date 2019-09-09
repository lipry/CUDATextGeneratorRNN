//
// Created by Fabio Lipreri on 2019-05-21.
//

#include "hyperbolic_tan.h"
#include "../utils/common.h"
#include "../utils/cudamath.h"


Matrix& Tanh::forward(Matrix &v) {
    this->V = v;
    R.allocate_size(v.getX(), v.getY());

    dim3 TxB(BLOCK_SIZE);
    dim3 num_blocks((V.getY() * V.getX() + TxB.x - 1) / TxB.x);
    tanhForward<<<num_blocks, TxB>>>(R.getDevData().get(), V.getDevData().get(), R.getX(), R.getY());

    return R;
}

Matrix& Tanh::backward(Matrix &top_diff) {
    dX.allocate_size(R.getX(), R.getY());

    dim3 TxB(BLOCK_SIZE);
    dim3 num_blocks((R.getY() * R.getX() + TxB.x - 1) / TxB.x);
    tanhBackward<<<num_blocks, TxB>>>(dX.getDevData().get(), R.getDevData().get(),
            top_diff.getDevData().get(), R.getX(), R.getY());

    return dX;
}

const Matrix &Tanh::getR() const {
    return R;
}

const Matrix &Tanh::getDx() const {
    return dX;
}
