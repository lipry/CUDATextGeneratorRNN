//
// Created by Fabio Lipreri on 2019-05-14.
//
#include <iostream>
#include "common.h"
#include "matrix.h"

Matrix::Matrix(size_t x, size_t y) : x(x), y(y), host_alloc(false), dev_alloc(false),
dev_data(nullptr), host_data(nullptr)
{}

void Matrix::allocHostMemory() {
    if(!host_alloc){
        host_data = std::shared_ptr<float>(new float[x*y], [&](float *p) {delete[] p; });
        if(host_data)
            host_alloc = true;
    }
}

void Matrix::allocDevMemory() {
    if(!dev_alloc){
        float* tmp = nullptr;
        CHECK(cudaMalloc((void **)&tmp, x*y*sizeof(float)));
        dev_data = std::shared_ptr<float>(tmp, [&](float *p){ cudaFree(p); });
        if(dev_data)
            dev_alloc = true;
    }
}

void Matrix::allocate_size(size_t x, size_t y) {
    if(!dev_alloc && !host_alloc) {
        this->x = x;
        this->y = y;
        allocHostMemory();
        allocDevMemory();
    }
}

void Matrix::allocate() {
    if(!dev_alloc && !host_alloc) {
        allocHostMemory();
        allocDevMemory();
    }
}


void Matrix::cpyHostToDev() {
    if(dev_alloc && host_alloc) {
        CHECK(cudaMemcpy(dev_data.get(), host_data.get(), x * y * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void Matrix::cpyDevToHost() {
    if(dev_alloc && host_alloc)
        CHECK(cudaMemcpy(host_data.get(), dev_data.get(), x*y*sizeof(float), cudaMemcpyDeviceToHost));
}

size_t Matrix::getX() const {
    return x;
}

size_t Matrix::getY() const {
    return y;
}

void Matrix::print_matrix() {
    for(int i = 0; i < x; i++){
        for(int j = 0; j < y; j++)
            printf("%f ", host_data.get()[i*y+j]);
        printf("\n");
    }
    printf("\n");
}

const std::shared_ptr<float> &Matrix::getHostData() const {
    return host_data;
}

const std::shared_ptr<float> &Matrix::getDevData() const {
    return dev_data;
}

float& Matrix::operator[](const int index) {
    return host_data.get()[index];
}

const float& Matrix::operator[](const int index) const {
    return host_data.get()[index];
}
