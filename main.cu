#include "src/operations/sigmoid.h"
#include "src/utils/matrix.h"

#include "cublas_v2.h"

#include "src/operations/hyperbolic_tan.h"
#include <math.h>
#include <stdio.h>
#include "src/utils/matrix.h"
#include "src/operations/add.h"
#include "src/utils/common.h"
#include "src/operations/prodmatvect.h"

#define N 4

void test_add();

int main(void) {
    /*cublasHandle_t handle;
    int len_x = 4;
    int len_y = 6;
    float alpha = 1.0f;
    float * x  = (float *) malloc(len_x * sizeof(float));
    float * y  = (float *) malloc(len_y * sizeof(float));
    float * res = (float *) malloc(len_x*len_y);

    float * d_x;
    float * d_y;
    float * d_res;

    cudaMalloc((void **) &d_x, sizeof(float) * len_x);
    cudaMalloc((void **) &d_y, sizeof(float) * len_y);
    cudaMalloc((void **) &d_res, sizeof(float) * len_x * len_y);

    int ctr = 0;
    for(int r = 0; r < len_x; r++){ //vettore colonna
            x[r] =  5;//ctr++
    }

    ctr = 0;
    for (int c=0; c < len_y; c++) {
        y[c] = 7;//ctr++;
    }

    cublasCreate(&handle);

    cublasSetVector(len_x, sizeof(float), x, 1, d_x, 1);
    cublasSetVector(len_y, sizeof(float), y, 1, d_y, 1);

    CHECK_CUBLAS(cublasSger(handle, len_x, len_y,
                            &alpha, d_x, 1, d_y, 1,
                            d_res, len_x));

    cublasGetMatrix(len_x, len_y, sizeof(float), d_res, len_x, res, len_x);

    printf("X\n");
    for(int r = 0; r < len_x; r++){ //vettore colonna
        printf("%f \n", x[r]);
    }
    printf("\n");

    printf("Y\n");
    for (int c=0; c < len_y; c++) {
        printf("%f ", y[c]); //ctr++;
    }
    printf("\n\n");


    printf("RES\n");
    for(int i = 0; i < len_x; i++){
        for(int j = 0; j < len_y; j++)
            printf("%f ", res[i*len_x+j]);
        printf("\n");
    }
    printf("\n");

    free(res);
    free(x);
    free(y);

    cudaFree(d_res);
    cudaFree(d_x);
    cudaFree(d_y);*/
    Matrix m = Matrix(N, N);
    Matrix v = Matrix(N, 1);
    Matrix top_diff = Matrix(N, 1);

    Matrix res;
    Matrix porcodio;
    Matrix dv;

    ProdMatVect pmv = ProdMatVect();
    m.allocate();
    v.allocate();
    top_diff.allocate();

    randimatrix(m, 5);
    randimatrix(v, 5);
    randimatrix(top_diff, 5);

    printf("M\n");
    m.print_matrix();
    printf("V\n");
    v.print_matrix();
    printf("TOP DIFF\n");
    top_diff.print_matrix();

    m.cpyHostToDev();
    v.cpyHostToDev();
    top_diff.cpyHostToDev();

    res = pmv.forward(m, v);

    res.cpyDevToHost();

    printf("FORWARD\n");
    res.print_matrix();

    pmv.backward(top_diff);
    porcodio = pmv.getdW();
    dv = pmv.getdv();

    dv.cpyDevToHost();
    porcodio.cpyDevToHost();

    printf("DW\n");
    porcodio.print_matrix();
    printf("Dv\n");
    dv.print_matrix();
    //test_add();
    cudaDeviceReset();
    return 0;
}

void test_add(){
    Matrix a = Matrix(N, N);
    Matrix b = Matrix(N, N);
    Matrix r;
    Matrix dX;

    Add x;

    srand (time(NULL));

    a.allocate();
    b.allocate();

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            a[i*N+j] = (float) (rand() % 100);
    }

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            b[i*N+j] = (float) (rand() % 100);
    }

    printf("W\n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            printf("%f ", a[i*N+j]);
        printf("\n");
    }
    printf("\n");

    printf("V\n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            printf("%f ", b[i*N+j]);
        printf("\n");
    }
    printf("\n");

    a.cpyHostToDev();
    b.cpyHostToDev();

    r = x.forward(a, b);
    r.cpyDevToHost();

    dX = x.backward(a);
    dX.cpyDevToHost();


    printf("SOMMA\n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            printf("%f ", r[i*N+j]);
        printf("\n");
    }
    printf("\n");

}
/*
void test_sigmoid(){
    Matrix m = Matrix(N, N);
    Matrix top_diff = Matrix(N, N);
    //Matrix r;
    Sigmoid s;

    m.allocate();
    top_diff.allocate();

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            m[i*N+j] = (float) i+j;
    }

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            top_diff[i*N+j] = (float) i-j;
    }

    printf("ORIGINALE\n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            printf("%f ", m[i*N+j]);
        printf("\n");
    }
    printf("\n");

    printf("TOP DIFF\n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            printf("%f ", top_diff[i*N+j]);
        printf("\n");
    }
    printf("\n");

    m.cpyHostToDev();
    m = s.forward(m);
    m.cpyDevToHost();

    printf("DOPO FORWARD\n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            printf("%f ", m[i*N+j]);
        printf("\n");
    }
    printf("\n");

    top_diff.cpyHostToDev();
    m = s.forward(top_diff);
    m.cpyDevToHost();

    printf("DOPO BACKWARD\n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            printf("%f ", m[i*N+j]);
        printf("\n");
    }
    printf("\n");
}
void test_tanh(){
    Matrix m = Matrix(N, N);
    Matrix top_diff = Matrix(N, N);
    //Matrix r;
    Tanh s;

    m.allocate();
    top_diff.allocate();

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            m[i*N+j] = (float) i+j;
    }

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            top_diff[i*N+j] = (float) i-j;
    }

    printf("ORIGINALE\n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            printf("%f ", m[i*N+j]);
        printf("\n");
    }
    printf("\n");

    printf("TOP DIFF\n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            printf("%f ", top_diff[i*N+j]);
        printf("\n");
    }
    printf("\n");

    m.cpyHostToDev();
    m = s.forward(m);
    m.cpyDevToHost();

    printf("DOPO FORWARD\n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            printf("%f ", m[i*N+j]);
        printf("\n");
    }
    printf("\n");

    top_diff.cpyHostToDev();
    m = s.backward(top_diff);
    m.cpyDevToHost();

    printf("DOPO BACKWARD\n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            printf("%f ", m[i*N+j]);
        printf("\n");
    }
    printf("\n");
}*/