//
// Created by Fabio Lipreri on 2019-05-31.
//

#ifndef PROGETTOGPU_PRODMATVECT_H
#define PROGETTOGPU_PRODMATVECT_H


#include "../utils/matrix.h"

class ProdMatVect {
private:
    Matrix M;
    Matrix V;
    Matrix R;
    Matrix dM;
    Matrix dv;

public:
    Matrix& forward(cublasHandle_t handle, Matrix& a, Matrix& v);
    void backward(cublasHandle_t handle, Matrix& top_diff);

    Matrix& getdMatrix();
    Matrix& getdVector();
};


#endif //PROGETTOGPU_PRODMATVECT_H
