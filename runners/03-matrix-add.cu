#include <KernelMatrixAdd.cuh>

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    int MATRIX_SIZE_CNT = std::atoi(argv[1]);
    int MATRIX_SIZE_BYTES = MATRIX_SIZE_CNT * sizeof(float);
    int BLOCK_SIZE = std::atoi(argv[2]);

    float* h_A = new float[MATRIX_SIZE_CNT * MATRIX_SIZE_CNT];
    float* h_B = new float[MATRIX_SIZE_CNT * MATRIX_SIZE_CNT];
    float* h_result = new float[MATRIX_SIZE_CNT * MATRIX_SIZE_CNT];

    size_t pitch_bytes;
    float* d_A;    
    cudaMallocPitch(&d_A, &pitch_bytes, MATRIX_SIZE_BYTES, MATRIX_SIZE_BYTES);
    float* d_B;
    cudaMallocPitch(&d_B, &pitch_bytes, MATRIX_SIZE_BYTES, MATRIX_SIZE_BYTES);
    float* d_result;
    cudaMallocPitch(&d_result, &pitch_bytes, MATRIX_SIZE_BYTES, MATRIX_SIZE_BYTES);

    size_t pitch = pitch_bytes / sizeof(float);
    for (int i = 0; i < MATRIX_SIZE_CNT; ++i) {
        for (int j = 0; j < MATRIX_SIZE_CNT; ++j) { 
            h_A[i * MATRIX_SIZE_CNT + j] = float((i+j) % 10) / 10;
            h_B[i * MATRIX_SIZE_CNT + j] = float((i+j) % 50) / 50;
        }
    }

    cudaMemcpy2D(d_A, pitch_bytes, h_A, MATRIX_SIZE_BYTES, MATRIX_SIZE_BYTES, MATRIX_SIZE_CNT, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_B, pitch_bytes, h_B, MATRIX_SIZE_BYTES, MATRIX_SIZE_BYTES, MATRIX_SIZE_CNT, cudaMemcpyHostToDevice);

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

	  int GRID_SIZE = (MATRIX_SIZE_CNT + BLOCK_SIZE - 1) / BLOCK_SIZE / 2 + 1;
	  KernelMatrixAdd<<<dim3(GRID_SIZE, GRID_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(
        MATRIX_SIZE_CNT, MATRIX_SIZE_CNT, pitch, d_A, d_B, d_result
    );

    cudaEventRecord(stop);

    cudaMemcpy2D(h_result, MATRIX_SIZE_BYTES, d_result, pitch_bytes, MATRIX_SIZE_BYTES, MATRIX_SIZE_CNT, cudaMemcpyDeviceToHost);

    for (int i = 0; i < std::min(5, MATRIX_SIZE_CNT); ++i) {
        std::cout << h_result[(MATRIX_SIZE_CNT-1-i) * MATRIX_SIZE_CNT] << ' ' << h_result[MATRIX_SIZE_CNT-1-i] << std::endl;
    }

    float milliseconds = 0.0;
	  cudaEventElapsedTime(&milliseconds, start, stop);
	  std::cout << "Time: " << milliseconds << " ms, size: " << MATRIX_SIZE_CNT << ", block: " << BLOCK_SIZE << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);

    delete [] h_A;
    delete [] h_B;
    delete [] h_result;

    return 0;
}