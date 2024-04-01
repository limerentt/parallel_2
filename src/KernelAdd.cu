#include "KernelAdd.cuh"


__global__ void KernelAdd(int numElements, float* x, float* y, float* result) {
    int start_index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = start_index; i < numElements; i += stride) {
        y[i] += x[i];
        result[i] = y[i];
    }
}