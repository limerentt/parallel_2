#include <ScalarMul.cuh>

/*
 * Calculates scalar multiplication for block
 */
__global__
void ScalarMulBlock(int numElements, float* vector1, float* vector2, float *result) {
    int start_index = threadIdx.x, stride = blockDim.x;

    result[start_index] = 0;
    for (int i = start_index; i < numElements; i += stride) {
        result[index] += vector1[i] * vector2[i];
    }
}