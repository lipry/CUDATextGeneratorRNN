
#include <iostream>
#include "matrix.h"
#include "common.h"


using namespace std;

void printfmatrix(Matrix x, string title){
    cout << title << endl;
    for(int r = 0; r < x.getX(); r++){
        for(int c = 0; c < x.getY(); c++)
            printf("%f ", x[r*x.getY()+c]);
        printf("\n");
    }
}

void randfmatrix(Matrix &x, int high, int low) {
    srand (time(NULL));
    for(int r = 0; r < x.getX(); r++){
        for(int c = 0; c < x.getY(); c++)
            x[r*x.getY()+c] = low + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(high-low)));
    }
}

void randimatrix(Matrix &x, int high) {
    srand (time(NULL));
    for(int r = 0; r < x.getX(); r++){
        for(int c = 0; c < x.getY(); c++)
            x[r*x.getY()+c] = (float) (rand() % high);
    }
}
