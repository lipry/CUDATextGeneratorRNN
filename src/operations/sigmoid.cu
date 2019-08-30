//
// Created by Fabio Lipreri on 2019-05-14.
//

#include <math.h>
#include "sigmoid.h"
#include "../utils/matrix.h"
#include "../utils/common.h"
#include "../utils/cudamath.h"
#include <stdlib.h>
#include <stdio.h>


Matrix& Sigmoid::forward(Matrix &v){
    this->V = v;
    R.allocate_size(v.getX(), v.getY());

    dim3 TxB(BLOCK_SIZE);
    dim3 num_blocks((V.getY() * V.getX() + TxB.x - 1) / TxB.x);
    sigmoidForward<<<num_blocks, TxB>>>(R.getDevData().get(), V.getDevData().get(), R.getX(), R.getY());

    return R;

}

Matrix& Sigmoid::backward(Matrix &top_diff) {
    dX.allocate_size(R.getX(), R.getY());

    dim3 TxB(BLOCK_SIZE);
    dim3 num_blocks((R.getY() * R.getX() + TxB.x - 1) / TxB.x);
    sigmoidBackward<<<num_blocks, TxB>>>(dX.getDevData().get(), R.getDevData().get(),
            top_diff.getDevData().get(), R.getX(), R.getY());

    return dX;

}
