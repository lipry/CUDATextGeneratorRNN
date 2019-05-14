//
// Created by Fabio Lipreri on 2019-05-14.
//

#include "matrix.h"
#include <iostream>

Matrix::Matrix(size_t x, size_t y) {
    this->x = x;
    this->y = y;
    this->host_alloc = false;
    this->dev_alloc = false;
}

void Matrix::allocHostMemory() {
    if(!host_alloc){
        host_data = std::shared_ptr<float>(new float[x*y], [&](float *p) {delete[] p; });
        if(host_data)
            host_alloc = true;
    }
}

void Matrix::allocDevMemory() {

}

void Matrix::allocate(bool checkAllocation) {
    allocHostMemory();
}

void Matrix::cpyHostToDev() {

}

void Matrix::cpyDevToHost() {

}

float& Matrix::operator[](const int index) {
    return host_data.get()[index];
}

const float& Matrix::operator[](const int index) const {
    return dev_data.get()[index];
}
