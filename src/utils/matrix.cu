//
// Created by Fabio Lipreri on 2019-05-14.
//
#include <iostream>
#include "cublas_v2.h"
#include "matrix.h"
#include "common.h"

Matrix::Matrix(size_t x, size_t y) : x(x), y(y), host_alloc(false), dev_alloc(false),
dev_data(nullptr), host_data(nullptr)
{}

void Matrix::allocHostMemory() {
    if(!host_alloc){
        host_data = (float *)malloc (x * y * sizeof (*host_data));
        if(host_data)
            host_alloc = true;
    }
}

void Matrix::allocDevMemory() {
    if(!dev_alloc){
        CHECK(cudaMalloc((void **)&dev_data, x*y*sizeof(*host_data)));
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

void Matrix::destroy() {
    if(dev_alloc && host_alloc) {
        cudaFree (dev_data);
        free(host_data);
        dev_data = nullptr;
        host_data = nullptr;
        printf("destroy");
    }
}

void Matrix::cpyHostToDev() {
    if(dev_alloc && host_alloc) {
        CHECK(cudaMemcpy(dev_data, host_data, x * y * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void Matrix::cpyDevToHost() {
    if(dev_alloc && host_alloc)
        CHECK(cudaMemcpy(host_data, dev_data, x*y*sizeof(float), cudaMemcpyDeviceToHost));
}

void Matrix::cpyHostToDevCublas(){
    if(dev_alloc && host_alloc){
        //m = x, n = y
        if(isVector()) {
            CHECK_CUBLAS(cublasSetVector(x, sizeof(float), host_data, 1, dev_data, 1));
        }else {
            CHECK_CUBLAS(cublasSetMatrix(x, y, sizeof(float), host_data, x, dev_data, y));
        }
    }

}

void Matrix::cpyDevToHostCublas(){
    if(dev_alloc && host_alloc){
        //m = x, n = y
        if(isVector()){
            CHECK_CUBLAS(cublasGetVector(x, sizeof(float), dev_data, 1, host_data, 1));
        }else{
            CHECK_CUBLAS(cublasGetMatrix(x, y, sizeof(float), dev_data, x, host_data, y));
        }
    }
}

size_t Matrix::getX() const {
    return x;
}

size_t Matrix::getY() const {
    return y;
}

bool Matrix::isVector(){
    // return getX() == 1; ???
    return getY() == 1;
}

void Matrix::print_matrix() {
    // TODO: column-major
    printf("x = %d y = %d\n", x, y);
    for(int i = 0; i < x; i++){
        for(int j = 0; j < y; j++)
            printf("%f ", host_data[i*y+j]);
        printf("\n");
    }
    printf("\n");
}

float* Matrix::getHostData() const {
    return host_data;
}

float* Matrix::getDevData() const {
    return dev_data;
}

float& Matrix::operator[](const int index) {
    return host_data[index];
}

const float& Matrix::operator[](const int index) const {
    return host_data[index];
}
