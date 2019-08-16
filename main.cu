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
#include "src/RNN/RnnLayer.h"

#define N 7
#define Neurons 6


int main(void) {
    Matrix x = Matrix(N, 1);
    Matrix h_prev = Matrix(Neurons, 1);
    Matrix U = Matrix(Neurons, N);
    Matrix W = Matrix(Neurons, Neurons);
    Matrix V = Matrix(N, Neurons);

    Matrix diffh = Matrix(Neurons, 1);
    Matrix dVproduct = Matrix(N, 1);

    x.allocate();
    h_prev.allocate();
    diffh.allocate();
    dVproduct.allocate();

    for(int i = 0; i < N; i++){
        x[i] = 0;
    }
    x[2] = 1;

    for(int i = 0; i < Neurons; i++){
        diffh[i] = i+1;
    }
    for(int i = 0; i < N; i++){
        dVproduct[i] = i+1;
    }



    for(int i = 0; i < Neurons; i++){
        h_prev[i] = 0;
    }

    U.allocate();
    for(int r = 0; r<Neurons; r++){
        for(int c = 0; c<N; c++){
            U[r*N+c] = c+1;
        }
    }

    W.allocate();
    for(int r = 0; r<Neurons; r++){
        for(int c = 0; c<Neurons; c++){
            W[r*Neurons+c] = c+1;
        }
    }

    V.allocate();
    for(int r = 0; r<N; r++){
        for(int c = 0; c<Neurons; c++){
            V[r*Neurons+c] = c+1;
        }
    }

    printf("X: \n");
    x.print_matrix();
    printf("h_prev: \n");
    h_prev.print_matrix();
    printf("U: \n");
    U.print_matrix();
    printf("W: \n");
    W.print_matrix();
    printf("V: \n");
    V.print_matrix();

    printf("diffh: \n");
    diffh.print_matrix();
    printf("dVProd: \n");
    dVproduct.print_matrix();

    diffh.cpyHostToDev();
    dVproduct.cpyHostToDev();

    x.cpyHostToDev();
    h_prev.cpyHostToDev();
    U.cpyHostToDev();
    W.cpyHostToDev();
    V.cpyHostToDev();

    RnnLayer rnnlayer = RnnLayer();
    //rnnlayer.forward(x, h_prev, U, W, V);
    rnnlayer.backward(x, h_prev, U, W, V, diffh, dVproduct);





    cudaDeviceReset();
    return 0;
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
 }*/