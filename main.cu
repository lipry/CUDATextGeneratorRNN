#include "src/operations/sigmoid.h"
#include "src/utils/matrix.h"

#include "cublas_v2.h"

#include "src/operations/hyperbolic_tan.h"
#include <math.h>
#include <stdio.h>
#include "src/utils/matrix.h"
#include "src/operations/add.h"
#include "src/utils/common.h"

#define N 4

void test_add();

int main(void) {
    //Matrix m = Matrix(N, N);
    //Matrix v = Matrix(N, 1);
    //Matrix res = Matrix(N, 1);

    float *d_m;
    float *m = (float *)malloc (N * N * sizeof (*m));
    float *d_v;
    float *v = (float *)malloc (N  * sizeof (*v));
    float *d_r;
    float *res = (float *)malloc (N * sizeof (*res));

    CHECK(cudaMalloc((void **)&d_m, N*N*sizeof(*m)));
    CHECK(cudaMalloc((void **)&d_v, N*1*sizeof(*v)));
    CHECK(cudaMalloc((void **)&d_r, N*1*sizeof(*res)));

    float alpha = 1.0f;
    float beta = 1.0f;

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    //m.allocate();
    //v.allocate();
    //res.allocate();

    srand (time(NULL));
    for(int r = 0; r < N; r++){
        for(int c = 0; c < N; c++)
            m[r*N+c] = r+c;
    }

    for(int r = 0; r < N; r++){
        for(int c = 0; c < 1; c++)
            v[r*1+c] = r+c;
    }
    //randimatrix(m, 5);
    //randimatrix(v, 5);
    printf("x = %d y = %d\n", N, N);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            printf("%f ", m[i*1+j]);
        printf("\n");
    }
    printf("\n");

    printf("x = %d y = %d\n", N, 1);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < 1; j++)
            printf("%f ", v[i*1+j]);
        printf("\n");
    }
    printf("\n");

    //m.print_matrix();
    //v.print_matrix();

    //m.cpyHostToDev();
    //v.cpyHostToDev();
    CHECK(cudaMemcpy(d_m, m, N*N*sizeof(*m), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_v, v, N*sizeof(*v), cudaMemcpyHostToDevice));

    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, N,
            N, &alpha, d_m, N, d_v, 1, &beta, d_r, 1));
    cudaThreadSynchronize();

    CHECK(cudaMemcpy(res, d_r, N*1*sizeof(float), cudaMemcpyDeviceToHost));
    //res.cpyDevToHost();
    //res.print_matrix();

    printf("x = %d y = %d\n", N, 1);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < 1; j++)
            printf("%f ", res[i*1+j]);
        printf("\n");
    }
    printf("\n");

    //m.destroy(); v.destroy();
    free(m);
    free(v);
    free(res);
    cudaFree(d_m);
    cudaFree(d_v);
    cudaFree(d_r);

    cublasDestroy(handle);

    printf("CUDA RESET");
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

    printf("A\n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            printf("%f ", a[i*N+j]);
        printf("\n");
    }
    printf("\n");

    printf("B\n");
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
            printf("%f ", dX[i*N+j]);
        printf("\n");
    }
    printf("\n");

}

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