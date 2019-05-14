//
// Created by Fabio Lipreri on 2019-05-14.
//
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

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

#endif //PROGETTOGPU_COMMON_H
