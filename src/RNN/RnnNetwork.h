//
// Created by Fabio Lipreri on 2019-09-09.
//

#ifndef PROGETTOGPU_RNNNETWORK_H
#define PROGETTOGPU_RNNNETWORK_H


#include "../utils/matrix.h"

class RnnNetwork {
private:
    int input_dim;
    int hidden_dim;
    Matrix U;
    Matrix W;
    Matrix V;
public:
    RnnNetwork(int input_dim, int hidden_dim);

    void init_weights();
    const Matrix &getU() const;
    const Matrix &getW() const;
    const Matrix &getV() const;
};



#endif //PROGETTOGPU_RNNNETWORK_H
