//
// Created by Fabio Lipreri on 2019-05-31.
//

#ifndef PROGETTOGPU_PRODMATVECT_H
#define PROGETTOGPU_PRODMATVECT_H


#include "../utils/matrix.h"

class ProdMatVect {
private:
    Matrix W;
    Matrix V;
    Matrix R;
    Matrix dW;
    Matrix dv;

public:
    Matrix& forward(Matrix& a, Matrix& v);
    void backward(Matrix& top_diff);

    Matrix& getdW();
    Matrix& getdv();
};


#endif //PROGETTOGPU_PRODMATVECT_H
