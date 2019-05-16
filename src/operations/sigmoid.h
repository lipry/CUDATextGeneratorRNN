//
// Created by Fabio Lipreri on 2019-05-14.
//

#ifndef PROGETTOGPU_SIGMOID_H
#define PROGETTOGPU_SIGMOID_H


#include "../utils/matrix.h"

class Sigmoid {
private:
    Matrix V;
    Matrix R;

    Matrix dX;
public:
    Matrix& forward(Matrix &v);
    Matrix& backward(Matrix &top_diff);
};


#endif //PROGETTOGPU_SIGMOID_H
