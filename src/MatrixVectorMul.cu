#include <MatrixVectorMul.cuh>


__global__ void MatrixVectorMul(int height, int width, float* matrix, float* vector, float* result) {
    int start_index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    result[start_index] = 0;
    for (int i = start_index; i < height; i += stride) {
        for (int j = 0; j < width; ++j) {
            result[i] += matrix[i * width + j] * vector[j];
        }
    }
}