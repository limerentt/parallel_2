#include <MatrixVectorMul.cuh>

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    int MATRIX_WIDTH = std::atoi(argv[1]);
    int BLOCK_SIZE = std::atoi(argv[2]);

    float* h_matrix = new float[BLOCK_SIZE * MATRIX_WIDTH];
    float* h_vector = new float[MATRIX_WIDTH];
    float* h_result = new float[BLOCK_SIZE];

    float* d_matrix;
    cudaMalloc(&d_matrix, BLOCK_SIZE * MATRIX_WIDTH * sizeof(float));
    float* d_vector;
    cudaMalloc(&d_vector, MATRIX_WIDTH * sizeof(float));
    float* d_result;
    cudaMalloc(&d_result, BLOCK_SIZE * sizeof(float));

    for (int i = 0; i < BLOCK_SIZE; ++i) {
        for (int j = 0; j < MATRIX_WIDTH; ++j) {
            h_matrix[i * MATRIX_WIDTH + j] = float((i+j) % 10) / 10;
        }
    }
    for (int i = 0; i < MATRIX_WIDTH; ++i) {
        h_vector[i] = float(i % 10) / 5;
    }

    cudaMemcpy(d_matrix, h_matrix, BLOCK_SIZE * MATRIX_WIDTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, h_vector, MATRIX_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

	int GRID_SIZE = 1;
	MatrixVectorMul<<<GRID_SIZE, BLOCK_SIZE>>>(BLOCK_SIZE, MATRIX_WIDTH, d_matrix, d_vector, d_result);

    cudaEventRecord(stop);

    cudaMemcpy(h_result, d_result, BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < std::min(5, BLOCK_SIZE); ++i) {
        std::cout << h_result[i] << ' ' << h_result[BLOCK_SIZE-1-i] << std::endl;
    }

    float milliseconds = 0.0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Time: " << milliseconds << " ms, size: " << MATRIX_WIDTH << ", block: " << BLOCK_SIZE << std::endl;

    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);

    delete [] h_matrix;
    delete [] h_vector;
    delete [] h_result;

    return 0;
}

