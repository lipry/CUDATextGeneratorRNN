#include <stdio.h>
#include "src/operations/sigmoid.h"
#include "src/operations/matrix.h"

/*
La “Formula” di base
1. Setup dei dati su host (CPU-accessible memory)
2. Alloca memoria per i dati sulla GPU
3. Copia i dati da host a GPU
4. Alloca memoria per output su host
5. Alloca memoria per output su GPU
6. Lancia il kernel su GPU
7. Copia output da GPU a host
8. Cancella le memorie
 */

#define N 4

int main(void) {
    Matrix m = Matrix(N, N);
    //Matrix r;
    Sigmoid s;

    m.allocate();

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            m[i*N+j] = (float) i+j;
    }

    /*for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            printf("%f ", m[i*N+j]);
        printf("\n");
    }
    printf("\n");*/

    m.cpyHostToDev();
    m = s.forward(m);
    m.cpyDevToHost();

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            printf("%f ", m[i*N+j]);
        printf("\n");
    }
    printf("\n");

    cudaDeviceReset();
    return 0;
}