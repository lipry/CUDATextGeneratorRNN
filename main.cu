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
    Matrix m = Matrix(N, N);
    Matrix v = Matrix(N, 1);
    Matrix top_diff = Matrix(N, 1);

    Matrix res;
    Matrix dW;
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
    dW = pmv.getdW();
    dv = pmv.getdv();

    dv.cpyDevToHost();
    dW.cpyDevToHost();

    printf("DW\n");
    dW.print_matrix();
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