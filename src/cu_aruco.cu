#include "cu_aruco.h"
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <vector>

namespace cu_aruco {

// Constructor implementation
CudaProcessor::CudaProcessor() {
    // Constructor implementation
}

bool CudaProcessor::InitCUDA()
{
	int count;

	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	int i;
	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) {
				break;
			}
		}
	}

	if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}

	cudaSetDevice(i);

	return true;
}

__global__ void threshold_kernel(const unsigned char* _src, unsigned char* _dst,
                                 int rows, int cols, int blockSize,
                                 double c) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols) {
	// for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < cols; j++) {
            int sum = 0;
            int count = 0;

            // Calculate local mean within the specified block size
            for (int x = -blockSize / 2; x <= blockSize / 2; x++) {
                for (int y = -blockSize / 2; y <= blockSize / 2; y++) {
                    int row = i + x;
                    int col = j + y;

                    // Ensure the pixel is within bounds
                    if (row >= 0 && row < rows && col >= 0 && col < cols) {
                        sum += _src[row * cols + col];
                        count++;
                    }
                }
            }

            int localMean = sum / count;
            int threshold = localMean - c;

            // Apply thresholding
            if (_src[i * cols + j] >= threshold) {
                _dst[i * cols + j] = 0; // Foreground
            } else {
                _dst[i * cols + j] = 255;   // Background
            }
    //     }
    // }
    }
}

void threshold_kernel_cpu(const unsigned char* _src, unsigned char* _mean, unsigned char* _dst,
                                 int rows, int cols, int blockSize,
                                 double c) {
	// printf("threshold_kernel_cpu!!!!!!\n");

	for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // int sum = 0;
            // int count = 0;

            // Calculate local mean within the specified block size
            // for (int x = -blockSize / 2; x <= blockSize / 2; x++) {
            //     for (int y = -blockSize / 2; y <= blockSize / 2; y++) {
            //         int row = i + x;
            //         int col = j + y;

            //         // Ensure the pixel is within bounds
            //         if (row >= 0 && row < rows && col >= 0 && col < cols) {
            //             sum += _src[row * cols + col];
            //             count++;
            //         }
            //     }
            // }

            // int localMean = sum / count;
            // int threshold = localMean - c;

            // Apply thresholding
            if (_src[i * cols + j] >= _mean[i * cols + j]) {
                _dst[i * cols + j] = 0; // Foreground
            } else {
                _dst[i * cols + j] = 255;   // Background
            }
        }
    }
}

static int iDivUp(int a, int b) { return (a%b != 0) ? (a/b + 1) : (a/b); }

// Implement the cuda_threshold method
void CudaProcessor::cuda_threshold(const unsigned char* _src, unsigned char* _dst, int rows, int cols, int step, int winSize, double constant) {

	dim3 blocks(iDivUp(rows, 16), iDivUp(cols, 16));
	dim3 threads(16, 16);
    
    // int threads = 128;  // You can adjust this based on your GPU's capability
    // int blocks = (rows * cols + threads - 1); // numThreadsPerBlock;

    // printf("cuda_threshold\n");


	// //=====================================================================================
	unsigned char* d_src;  // Device memory for _src
    unsigned char* d_dst;  // Device memory for _dst

	size_t dataSize = rows * cols * sizeof(unsigned char);

	cudaMalloc((void**)&d_src, dataSize);
    cudaMalloc((void**)&d_dst, dataSize);

	cudaMemcpy(d_src, _src, dataSize, cudaMemcpyHostToDevice);

	threshold_kernel<<<blocks, threads>>>(d_src, d_dst, rows, cols, winSize, constant);
	// threshold_kernel<<<1, 1>>>(d_src, d_dst, rows, cols, winSize, constant);

	cudaMemcpy(_dst, d_dst, dataSize, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
	//=====================================================================================
}

void CudaProcessor::cuda_threshold_preFilter(const unsigned char* _src, unsigned char* _mean, unsigned char* _dst, int rows, int cols, int step, int winSize, double constant) {
    threshold_kernel_cpu(_src, _mean, _dst, rows, cols, winSize, constant);
}

} // namespace cu_aruco