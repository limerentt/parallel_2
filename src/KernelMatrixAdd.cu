#include <KernelMatrixAdd.cuh>


__global__ void KernelMatrixAdd(int height, int width, int pitch, float* A, float* B, float* result) {
  int start_index = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int numElements = height * width;
  for (int i = start_index; i < numElements; i += stride) {
      int padded_index = pitch * (i / width) + i % width;
      result[padded_index] = A[padded_index] + B[padded_index];
  }
}