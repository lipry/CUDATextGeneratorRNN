//
// Created by Fabio Lipreri on 2019-06-14.
//

#ifndef PROGETTOGPU_RNNLAYER_H
#define PROGETTOGPU_RNNLAYER_H


#include "../operations/prodmatvect.h"
#include "../operations/add.h"
#include "../operations/sigmoid.h"

class RnnLayer {
private:
    ProdMatVect Uproduct;
    ProdMatVect Wproduct;
    Add UWsum;
    Sigmoid ht;
    ProdMatVect Vhproduct;
public:

};


#endif //PROGETTOGPU_RNNLAYER_H
