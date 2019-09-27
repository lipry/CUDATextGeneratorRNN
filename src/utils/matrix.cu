//
// Created by Fabio Lipreri on 2019-05-14.
//
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include "../utils/cudamath.h"
#include "cublas_v2.h"
#include "matrix.h"
#include "common.h"
#include <bits/stdc++.h>

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

bool Matrix::isDevAlloc(){
    return dev_alloc;
}

bool Matrix::isHostAlloc(){
    return host_alloc;
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

void Matrix::cpyHostToDevCublas(){
    if(dev_alloc && host_alloc){
        //m = x, n = y
        if(getY() == 1) {
            printf("entro primo");
            CHECK_CUBLAS(cublasSetVector(x, sizeof(float), host_data.get(), 1, dev_data.get(), 1));
        }else if(getX() == 1){
            printf("entro secondo");
            CHECK_CUBLAS(cublasSetVector(y, sizeof(float), host_data.get(), 1, dev_data.get(), 1));
        }else {
            CHECK_CUBLAS(cublasSetMatrix(x, y, sizeof(float), host_data.get(), x, dev_data.get(), y));
        }
    }

}

void Matrix::cpyDevToHostCublas(){
    if(dev_alloc && host_alloc){
        //m = x, n = y
        if(getY() == 1) {
            CHECK_CUBLAS(cublasGetVector(x, sizeof(float), dev_data.get(), 1, host_data.get(), 1));
        }else if(getX() == 1){
            CHECK_CUBLAS(cublasGetVector(y, sizeof(float), dev_data.get(), 1, host_data.get(), 1));
        }else{
            CHECK_CUBLAS(cublasGetMatrix(x, y, sizeof(float), dev_data.get(), x, host_data.get(), y));
        }
    }
}

size_t Matrix::getX() const {
    return x;
}

size_t Matrix::getY() const {
    return y;
}

void Matrix::matrix_like(float number, Matrix &mat){
    allocate_size(mat.getY(), mat.getX());
    for(int i = 0; i<getX()*getY(); i++){
        this->host_data.get()[i] = number;
    }
}

bool Matrix::isVector(){
    // return getX() == 1; ???
    return getY() == 1 || getX() == 1;
}

void Matrix::print_matrix() {
    // TODO: column-major
    printf("x = %d y = %d\n", x, y);
    for(int i = 0; i < x; i++){
        for(int j = 0; j < y; j++)
            printf("%f ", host_data.get()[i*y+j]);
        printf("\n");
    }
    printf("\n");
}

void Matrix::load_rand(float lower, float higher) {
    curandState_t* states;
    int N = getX()*getY();
    /* allocate space on the GPU for the random states */
    cudaMalloc((void**) &states, N * sizeof(curandState_t));

    dim3 TxB(BLOCK_SIZE);
    dim3 num_blocks((N + TxB.x - 1) / TxB.x);
    init_randoms<<<num_blocks, TxB>>>(time(0), states);
    randoms<<<num_blocks, TxB>>>(states, getDevData().get(), lower, higher);
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
