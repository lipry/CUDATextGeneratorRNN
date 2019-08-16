//
// Created by Fabio Lipreri on 2019-05-21.
//

#ifndef PROGETTOGPU_HYPERBOLIC_TAN_H
#define PROGETTOGPU_HYPERBOLIC_TAN_H

#include "../utils/matrix.h"

class Tanh {
private:
    Matrix V;
    Matrix R;

    Matrix dX;
public:
    Matrix& forward(Matrix &v);
    Matrix& backward(Matrix &top_diff);

    const Matrix &getR() const;

    const Matrix &getDx() const;
};


#endif //PROGETTOGPU_HYPERBOLIC_TAN_H
