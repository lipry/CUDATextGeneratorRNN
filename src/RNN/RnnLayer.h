//
// Created by Fabio Lipreri on 2019-06-14.
//

#ifndef PROGETTOGPU_RNNLAYER_H
#define PROGETTOGPU_RNNLAYER_H


#include "../operations/prodmatvect.h"
#include "../operations/add.h"
#include "../operations/hyperbolic_tan.h"

class RnnLayer {
private:
    ProdMatVect Uproduct;
    ProdMatVect Wproduct;
    Add UWsum;
    Tanh ht;
    ProdMatVect Vhproduct;
public:
    void forward(Matrix &x, Matrix &h_prev, Matrix &U, Matrix &W, Matrix &V);

};


#endif //PROGETTOGPU_RNNLAYER_H
