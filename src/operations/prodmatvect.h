//
// Created by Fabio Lipreri on 2019-05-31.
//

#ifndef PROGETTOGPU_PRODMATVECT_H
#define PROGETTOGPU_PRODMATVECT_H


#include "../utils/matrix.h"

class ProdMatVect {
private:
    Matrix A;
    Matrix B;
    Matrix R;
    Matrix dX;

public:
    Matrix& forward(Matrix& a, Matrix& b);
    Matrix& backward(Matrix& top_diff);
};


#endif //PROGETTOGPU_PRODMATVECT_H
