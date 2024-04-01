#include <CosineVector.cuh>

#include <iostream>
#include <cstdlib> 

int main(int argc, char** argv) {
    int VECTOR_SIZE_CNT = std::atoi(argv[1]);
    int VECTOR_SIZE_BYTES = VECTOR_SIZE_CNT * sizeof(float);
    int BLOCK_SIZE = std::atoi(argv[2]);

    float* h_x = new float[VECTOR_SIZE_CNT];
    float* h_y = new float[VECTOR_SIZE_CNT];
    for (int i = 0; i < VECTOR_SIZE_CNT; ++i) {
        h_x[i] = float(i % 10) / 5;
        h_y[i] = float(i % 50) / 10;
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    float cos = CosineVector(VECTOR_SIZE_CNT, h_x, h_y, BLOCK_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    std::cout << "Cosinus: " << cos << std::endl;
    float milliseconds = 0.0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Time: " << milliseconds << " ms, size: " << VECTOR_SIZE_CNT << ", block: " << BLOCK_SIZE << std::endl;

    delete [] h_x;
    delete [] h_y;

    return 0;
}