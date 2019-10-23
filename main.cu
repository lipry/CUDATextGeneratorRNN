#include "src/utils/matrix.h"
#include "src/operations/sigmoid.h"
#include "cublas_v2.h"
#include "src/operations/hyperbolic_tan.h"
#include <stdio.h>
#include "src/operations/add.h"
#include "src/utils/common.h"
#include "src/operations/prodmatvect.h"
#include "src/RNN/RnnLayer.h"
#include "src/utils/cudamath.h"
#include <cmath>
#include "src/operations/outputlayer.h"
#include "src/RNN/RnnNetwork.h"
#include <chrono>
#include <iostream>
#include <vector>


#define N 7
#define Neurons 6

//istanze N = 5000, Neurons = 5000, un loop di backpropagation su python 600ms, con cuda 5 ms.cx

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    //Matrix x = Matrix(N, 1);
    /*Matrix h_prev = Matrix(Neurons, 1);
    Matrix U = Matrix(Neurons, N);
    Matrix W = Matrix(Neurons, Neurons);
    Matrix V = Matrix(N, Neurons);

    Matrix diffh = Matrix(Neurons, 1);
    Matrix dVproduct = Matrix(N, 1);*/

    /*x.allocate();
    x.oneHotEncoder(2);
    x.print_matrix();*/

    /*Matrix M = Matrix(4, 3);
    M.allocate();
    int count = 1;
    for(int r = 0; r<4; r++){
        for(int c = 0; c<3; c++){
            M[r*3+c] = count;
            count++;
        }
    }
    M.cpyHostToDev();
    M.print_matrix();



    Matrix dz = Matrix(3, 1);
    dz.allocate();
    dz[0] = 1;
    dz[1] = 2;
    dz[2] = 3;
    dz[3] = 4;
    dz.cpyHostToDev();*/

    //float alpha = 1.0f;
    //float beta = 0.0f;

    /*size_t m = M.getY();
    size_t n = M.getX();
    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, m, n,
                             &alpha, M.getDevData().get(), m, x.getDevData().get(), 1, &beta, out.getDevData().get(), 1));*/

    /*ProdMatVect prd = ProdMatVect();
    Matrix mul = prd.forward(handle, M, x);
    prd.backward(handle, dz);

    Matrix D = prd.getdMatrix();
    Matrix v = prd.getdVector();

    mul.cpyDevToHost();
    cout << "Mul: " << mul << endl;


    D.cpyDevToHost();
    printf("D: \n");
    D.print_matrix();

    v.cpyDevToHost();
    printf("v: \n");
    v.print_matrix();*/
    std::vector<int> v{ 2, 3, 2, 3, 4, 5, 6, 6, 6};
    std::vector<int> y{ 1, 2, 3, 4, 5, 6, 6, 6, 6};
    RnnNetwork rnn = RnnNetwork(N, Neurons);
    //const vector<RnnLayer> &layers = rnn.forward_prop(handle, v);
    rnn.backprop_through_time(handle, v, y, 3);

    //for(int i = 0; i < layers.size(); i++){
    //    cout << layers[i] << endl;
    //}

    /*Matrix U = rnn.getU();
    Matrix W = rnn.getW();
    Matrix V = rnn.getV();


    U.cpyDevToHost();
    W.cpyDevToHost();
    V.cpyDevToHost();

    printf("U: \n");
    U.print_matrix();
    printf("W: \n");
    W.print_matrix();
    printf("V: \n");
    V.print_matrix();*/



    //x.allocate();

    /*h_prev.allocate();
    diffh.allocate();
    dVproduct.allocate();*/

    /*for(int i = 0; i < N; i++){
        x[i] = 0.0;
    }
    x[2] = 1.0;
    cout << x;*/

    /*for(int i = 0; i < Neurons; i++){
        diffh[i] = i+1;
    }

    float step = 0.3f/N;
    float val = 0.0f;
    for(int i = 0; i < N; i++){
        dVproduct[i] = val;
        val += step;
    }


    step = 1.0f/Neurons;
    val = 0.0f;
    for(int i = 0; i < Neurons; i++){
        h_prev[i] = val;
        val += step;
    }

    U.allocate();
    for(int r = 0; r<Neurons; r++){
        step = 1.0f/N;
        val = 0.0f;
        for(int c = 0; c<N; c++){
            U[r*N+c] = val;
            val += step;
        }
    }

    W.allocate();
    for(int r = 0; r<Neurons; r++){
        step = 0.5f/Neurons;
        val = 0.0f;
        for(int c = 0; c<Neurons; c++){
            W[r*Neurons+c] = val;
            val += step;
        }
    }

    V.allocate();
    for(int r = 0; r<N; r++){
        step = 0.2f/Neurons;
        val = 0.8f;
        for(int c = 0; c<Neurons; c++){
            V[r*Neurons+c] = val;
            val += step;
        }
    }

    /*printf("X: \n");
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
    dVproduct.print_matrix();*/

    /*diffh.cpyHostToDev();
    dVproduct.cpyHostToDev();

    x.cpyHostToDev();
    h_prev.cpyHostToDev();
    U.cpyHostToDev();
    W.cpyHostToDev();
    V.cpyHostToDev();



    RnnLayer rnnlayer;
    rnnlayer = RnnLayer();
    auto t1 = std::chrono::high_resolution_clock::now();
    rnnlayer.forward(handle, x, h_prev, U, W, V);
    rnnlayer.backward(handle, x, h_prev, U, W, V, diffh, dVproduct);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

    cout << "time elapsed: "<< duration / 1 << endl;

    Matrix h = rnnlayer.getH();
    Matrix output = rnnlayer.getOutput();
    Matrix dx = rnnlayer.getDx();
    Matrix dW = rnnlayer.getDW();
    Matrix dV = rnnlayer.getDV();
    Matrix dU = rnnlayer.getDU();
    Matrix dh_prev = rnnlayer.getDhPrev();

    h.cpyDevToHost();
    output.cpyDevToHost();
    dx.cpyDevToHost();
    dW.cpyDevToHost();
    dV.cpyDevToHost();
    dU.cpyDevToHost();
    dh_prev.cpyDevToHost();*/

    /*printf("h: \n");
    h.print_matrix();
    printf("output: \n");
    output.print_matrix();
    printf("dx \n");
    dx.print_matrix();
    printf("dW \n");
    dW.print_matrix();
    printf("dV \n");
    dV.print_matrix();
    printf("dU \n");
    dU.print_matrix();
    printf("dh_prev \n");
    dh_prev.print_matrix();*/
    cublasDestroy(handle);
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

    printf("M\n");
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