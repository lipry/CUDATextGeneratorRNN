//
// Created by Fabio Lipreri on 2019-05-06.
//
#include <stdio.h>
#define INDEX(rows, cols, stride) (rows*stride + cols)
#define BDIMY 3
#define BDIMX 3

__global__ void add_vect(float *a, float *b, float *result, int n){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx < n)
        result[idx] = a[idx] + b[idx];
}

__global__ void transpose_matrix(float *in, float *result, int nrows, int ncols){
    __shared__ float tile[BDIMY][BDIMX];

    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int offset = INDEX(row, col, ncols);

    if (row < nrows && col < ncols)
        tile[threadIdx.y][threadIdx.x] = in[offset];

    __syncthreads();

    unsigned int bidx, irow, icol;
    bidx = threadIdx.y * blockDim.x + threadIdx.x;
    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    col = blockIdx.y * blockDim.y + icol;
    row = blockIdx.x * blockDim.x + irow;

    unsigned int transposed_offset = INDEX(row, col, nrows);

    if (row < ncols && col < nrows)
        result[transposed_offset] = tile[icol][irow];
}
