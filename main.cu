#include <stdio.h>

__global__ void helloFromGPU (void) {
    int a = 0;
    printf("Hello from GPU!\n");
    printf("Max è scemo");
}

int main(void) {
    // hello from GPU
    int b = 0;
    printf("Hello World from CPU!\n");
    helloFromGPU <<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}