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

// __global__ void threshold_kernel(const unsigned char* _src, unsigned char* _dst,
//                                  int rows, int cols, int blockSize, double constant) {

//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;

//     // Shared memory for the block
//     __shared__ int sharedBlock[32][32];  // Assuming maximum block size is 32x32

//     if (i < rows && j < cols) {

//         int local_i = threadIdx.x;
//         int local_j = threadIdx.y;

//         int sum = 0;
//         int count = 0;

//         // Load data into shared memory
//         sharedBlock[local_i][local_j] = _src[i * cols + j];
//         __syncthreads();

//         // Calculate local mean within the specified block size
//         for (int x = -blockSize / 2; x <= blockSize / 2; x++) {
//             for (int y = -blockSize / 2; y <= blockSize / 2; y++) {
//                 int row = local_i + x;
//                 int col = local_j + y;

//                 // Ensure the pixel is within bounds
//                 if (row >= 0 && row < blockDim.x && col >= 0 && col < blockDim.y) {
//                     sum += sharedBlock[row][col];
//                     count++;
//                 }
//             }
//         }

//         int localMean = sum / count;
//         int threshold = localMean - constant;

//         // Apply thresholding
//         _dst[i * cols + j] = (_src[i * cols + j] >= threshold) ? 0 : 255;

//     }
// }

// __global__ void threshold_kernel(const unsigned char* _src, unsigned char* _dst,
//                                  int rows, int cols, int blockSize, double constant) {

//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;

//     if (i < rows && j < cols) {

//         int sum = 0;
//         int count = 0;

//         // Calculate local mean within the specified block size
//         for (int x = -blockSize / 2; x <= blockSize / 2; x++) {
//             for (int y = -blockSize / 2; y <= blockSize / 2; y++) {
//                 int row = i + x;
//                 int col = j + y;

//                 // Ensure the pixel is within bounds
//                 if (row >= 0 && row < rows && col >= 0 && col < cols) {
//                     sum += _src[row * cols + col];
//                     count++;
//                 }
//             }
//         }

//         int localMean = sum / count;
//         int threshold = localMean - constant;

//         // Apply thresholding
//         if (_src[i * cols + j] >= threshold) {
//             _dst[i * cols + j] = 0; // Foreground
//         } else {
//             _dst[i * cols + j] = 255; // Background
//         }
//     }
// }

__global__ void calculateMeanRows(const unsigned char* _src, int* _mean, int rows, int cols, int blockSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int blockSizes[3] = {13, 23, 33};

    if (i < rows && j < cols) {
        int sum = 0;
        int count = 0;

        for (int bS = 0; bS < 3; bS++) {
            for (int x = -blockSizes[bS] / 2; x <= blockSizes[bS] / 2; x++) {
                int col = j + x;

                if (col >= 0 && col < cols 
                    && (bS == 0 || (x < -blockSizes[bS - 1] / 2) || (x > blockSizes[bS - 1] / 2))) {
                    sum += _src[i * cols + col];
                    count++;
                }
            }
            _mean[(i * cols + j) + bS * rows * cols] = sum / count;
        }
    }
}

__global__ void calculateMeanCols(const int* _meanRows, int* _mean, int rows, int cols, int blockSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int blockSizes[3] = {13, 23, 33};

    if (i < rows && j < cols) {
        int sum = 0;
        int count = 0;
        
        for (int bS = 0; bS < 3; bS++) {
            for (int y = -blockSizes[bS] / 2; y <= blockSizes[bS] / 2; y++) {
                int row = i + y;

                if (row >= 0 && row < rows 
                    && (bS == 0 || (y < -blockSizes[bS - 1] / 2) || (y > blockSizes[bS - 1] / 2))) {
                    sum += _meanRows[row * cols + j + bS * rows * cols];
                    count++;
                }
            }

            _mean[(i * cols + j) + bS * rows * cols] = sum / count;
        }
    }
}

// __global__ void applyThreshold(const unsigned char* _src, unsigned char* _dst, const int* _mean, int rows, int cols, double c) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;

//     if (i < rows && j < cols) {
//         int threshold = _mean[i * cols + j] - c;

//         // Apply thresholding
//         _dst[i * cols + j] = (_src[i * cols + j] >= threshold) ? 0 : 255;
//     }
// }


__global__ void applyThreshold_n(const unsigned char* _src, unsigned char* _dst, int* _mean, int rows, int cols, double c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols) {
        for (int k = 0; k < 3; ++k) {
            int threshold = _mean[(i * cols + j) + k * rows * cols] - c;
            _dst[k * rows * cols + i * cols + j] = (_src[i * cols + j] >= threshold) ? 0 : 255;
        }
    }
} 

void CudaProcessor::codaMalloc_space_for_image(unsigned char*& d_src, unsigned char*& d_dst, size_t dataSize){
    cudaMalloc((void**)&d_src, dataSize);
    // cudaMalloc((void**)&d_dst, dataSize);
    cudaMalloc((void**)&d_dst, dataSize*3);//for testinggggggggggggggggggggggg!!!
}

void CudaProcessor::update_image_to_VRAM(unsigned char* _src, unsigned char* _dst, unsigned char* d_src, unsigned char* d_dst, size_t dataSize){
    cudaMemcpy(d_src, _src, dataSize, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_dst, _dst, dataSize, cudaMemcpyHostToDevice);
}

void CudaProcessor::download_image_from_VRAM(unsigned char* _dst, unsigned char* d_dst, size_t dataSize){

    cudaMemcpy(_dst, d_dst, dataSize, cudaMemcpyDeviceToHost);
}

void CudaProcessor::free_up_VRAM(unsigned char* d_src, unsigned char* d_dst){
    cudaFree(d_src);
    cudaFree(d_dst);
}

static int iDivUp(int a, int b) { return (a%b != 0) ? (a/b + 1) : (a/b); }

void CudaProcessor::cuda_threshold_n(const unsigned char* d_src, unsigned char* d_dst, int rows, int cols, int step, int winSize, double constant, int nScale) {
    // printf("cuda_threshold_GPU_n %d\n", winSize);

	dim3 blocks(iDivUp(rows, 16), iDivUp(cols, 16));
	dim3 threads(16, 16);

    // // Calculate the addresses for d_dst1, d_dst2, and d_dst3
    size_t dataSize = cols * rows * sizeof(unsigned char);

    int* d_mean;
    int* d_meanRows;
    size_t meanSize = rows * cols * sizeof(int);//TODO
    cudaMalloc((void**)&d_meanRows, meanSize * 3);
    cudaMalloc((void**)&d_mean, meanSize * 3);

    calculateMeanRows<<<blocks, threads>>>(d_src, d_meanRows, rows, cols, winSize);
    calculateMeanCols<<<blocks, threads>>>(d_meanRows, d_mean, rows, cols, winSize);

    applyThreshold_n<<<blocks, threads>>>(d_src, d_dst, d_mean, rows, cols, constant);

    cudaDeviceSynchronize(); // Wait for all kernels to finish

    cudaFree(d_mean);
    cudaFree(d_meanRows);
}

    


} // namespace cu_aruco