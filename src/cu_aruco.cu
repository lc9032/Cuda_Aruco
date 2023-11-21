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
                                 int rows, int cols, int blockSize, double constant) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols) {

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
        int threshold = localMean - constant;

        // Apply thresholding
        if (_src[i * cols + j] >= threshold) {
            _dst[i * cols + j] = 0; // Foreground
        } else {
            _dst[i * cols + j] = 255; // Background
        }
    }
}

// __global__ void calculateMean(const unsigned char* _src, int* _mean, int rows, int cols, int blockSize) {
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

//         _mean[i * cols + j] = sum / count;
//     }
// }

// __global__ void applyThreshold(const unsigned char* _src, unsigned char* _dst, const int* _mean, int rows, int cols, double c) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;

//     if (i < rows && j < cols) {
//         int threshold = _mean[i * cols + j] - c;

//         // Apply thresholding
//         if (_src[i * cols + j] >= threshold) {
//             _dst[i * cols + j] = 0; // Foreground
//         } else {
//             _dst[i * cols + j] = 255; // Background
//         }

//         // _dst[i * cols + j] = (_src[i * cols + j] >= threshold) ? 0 : 255;
//     }
// }

void threshold_kernel_cpu(const unsigned char* _src, unsigned char* _dst,
                                 int rows, int cols, int blockSize,
                                 double c) {
	// printf("threshold_kernel_cpu!!!!!!\n");

	for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
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
            if (_src[i * cols + j] >= localMean) {
                _dst[i * cols + j] = 0; // Foreground
            } else {
                _dst[i * cols + j] = 255; // Background
            }
        }
    }
}

void CudaProcessor::codaMalloc_space_for_image(unsigned char*& d_src, unsigned char*& d_dst, size_t dataSize){
    cudaMalloc((void**)&d_src, dataSize);
    cudaMalloc((void**)&d_dst, dataSize);
}

void CudaProcessor::update_image_to_VRAM(unsigned char* _src, unsigned char* _dst, unsigned char* d_src, unsigned char* d_dst, size_t dataSize){
    cudaMemcpy(d_src, _src, dataSize, cudaMemcpyHostToDevice);

    cudaMemcpy(d_dst, _dst, dataSize, cudaMemcpyHostToDevice);//
}

void CudaProcessor::download_image_from_VRAM(unsigned char* _dst, unsigned char* d_dst, size_t dataSize){

    cudaMemcpy(_dst, d_dst, dataSize, cudaMemcpyDeviceToHost);
}

void CudaProcessor::free_up_VRAM(unsigned char* d_src, unsigned char* d_dst){
    cudaFree(d_src);
    cudaFree(d_dst);
}

#if CPU_GPU_SWITCH == 0

void CudaProcessor::cuda_threshold(const unsigned char* _src, unsigned char* _dst, int rows, int cols, int step, int winSize, double constant) {
    // printf("cuda_threshold_CPU\n");

    threshold_kernel_cpu(_src, _dst, rows, cols, winSize, constant);
}

#else

static int iDivUp(int a, int b) { return (a%b != 0) ? (a/b + 1) : (a/b); }

// Implement the cuda_threshold method
void CudaProcessor::cuda_threshold(const unsigned char* d_src, unsigned char* d_dst, int rows, int cols, int step, int winSize, double constant) {
    // printf("cuda_threshold_GPU\n");

	dim3 blocks(iDivUp(rows, 16), iDivUp(cols, 16));
	dim3 threads(16, 16);

    threshold_kernel<<<blocks, threads>>>(d_src, d_dst, rows, cols, winSize, constant);

    // int* d_mean;
    // size_t dataSize = rows * cols * sizeof(unsigned char);
    // cudaMalloc((void**)&d_mean, dataSize);

    // calculateMean<<<blocks, threads>>>(d_src, d_mean, rows, cols, winSize);

    // applyThreshold<<<blocks, threads>>>(d_src, d_dst, d_mean, rows, cols, constant);

}
#endif
    


} // namespace cu_aruco