#include "KernelMul.cuh"

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    int VECTOR_SIZE_CNT = std::atoi(argv[1]);
    int VECTOR_SIZE_BYTES = VECTOR_SIZE_CNT * sizeof(float);
    int BLOCK_SIZE = std::atoi(argv[2]);

    float* h_x = new float[VECTOR_SIZE_CNT];
    float* h_y = new float[VECTOR_SIZE_CNT];
    float* h_result = new float[VECTOR_SIZE_CNT];

    float* d_x;
    cudaMalloc(&d_x, VECTOR_SIZE_BYTES);
    float* d_y;
    cudaMalloc(&d_y, VECTOR_SIZE_BYTES);
    float* d_result;
    cudaMalloc(&d_result, VECTOR_SIZE_BYTES);

    for (int i = 0; i < VECTOR_SIZE_CNT; ++i) {
        h_x[i] = float(i % 10) / 5;
        h_y[i] = float(i % 50) / 10;
    }

    cudaMemcpy(d_x, h_x, VECTOR_SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, VECTOR_SIZE_BYTES, cudaMemcpyHostToDevice);

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

	int GRID_SIZE = (VECTOR_SIZE_CNT + BLOCK_SIZE - 1) / BLOCK_SIZE / 2 + 1;
	KernelMul<<<GRID_SIZE, BLOCK_SIZE>>>(VECTOR_SIZE_CNT, d_x, d_y, d_result);

    cudaEventRecord(stop);

    cudaMemcpy(h_result, d_result, VECTOR_SIZE_BYTES, cudaMemcpyDeviceToHost);

    for (int i = 0; i < std::min(5, VECTOR_SIZE_CNT); ++i) {
        std::cout << h_result[i] << ' ' << h_result[VECTOR_SIZE_CNT-1-i] << std::endl;
    }

    float milliseconds = 0.0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Time: " << milliseconds << " ms, size: " << VECTOR_SIZE_CNT << ", block: " << BLOCK_SIZE << std::endl;

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);

    delete [] h_x;
    delete [] h_y;
    delete [] h_result;

    return 0;
}