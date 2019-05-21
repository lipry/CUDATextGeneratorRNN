//
// Created by Fabio Lipreri on 2019-05-14.
//

#ifndef PROGETTOGPU_ADD_H
#define PROGETTOGPU_ADD_H


#include "../utils/matrix.h"

class Add{
private:
    Matrix A;
    Matrix B;
    Matrix R;
    Matrix dX;

public:
    Matrix& forward(Matrix& a, Matrix& b);
    Matrix& backward(Matrix& top_diff);
};


#endif //PROGETTOGPU_ADD_H
