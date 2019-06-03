//
// Created by Fabio Lipreri on 2019-05-14.
//
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

using namespace std;

#ifndef PROGETTOGPU_COMMON_H
#define PROGETTOGPU_COMMON_H

#define BLOCK_SIZE 256

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

void printfmatrix(Matrix x, string title);
void randfmatrix(Matrix& x, int high, int low);
void randimatrix(Matrix& x, int high);


#endif //PROGETTOGPU_COMMON_H
