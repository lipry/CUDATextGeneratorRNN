//
// Created by Fabio Lipreri on 2019-09-05.
//

#ifndef PROGETTOGPU_OUTPUTLAYER_H
#define PROGETTOGPU_OUTPUTLAYER_H


#include "../utils/matrix.h"
#include "cublas_v2.h"

class OutputLayer {
private:
    Matrix predictions;
public:
    void predict(cublasHandle_t handle, const Matrix &x);
    float loss(cublasHandle_t handle, const Matrix &x, int y);
    const Matrix & diff(cublasHandle_t handle, const Matrix &x, int y);
    const Matrix &getPredictions() const;
};


#endif //PROGETTOGPU_OUTPUTLAYER_H
