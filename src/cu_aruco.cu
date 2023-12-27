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

    // Shared memory for the block
    __shared__ int sharedBlock[32][32];  // Assuming maximum block size is 32x32

    if (i < rows && j < cols) {

        int local_i = threadIdx.x;
        int local_j = threadIdx.y;

        int sum = 0;
        int count = 0;

        // Load data into shared memory
        sharedBlock[local_i][local_j] = _src[i * cols + j];
        __syncthreads();

        // Calculate local mean within the specified block size
        for (int x = -blockSize / 2; x <= blockSize / 2; x++) {
            for (int y = -blockSize / 2; y <= blockSize / 2; y++) {
                int row = local_i + x;
                int col = local_j + y;

                // Ensure the pixel is within bounds
                if (row >= 0 && row < blockDim.x && col >= 0 && col < blockDim.y) {
                    sum += sharedBlock[row][col];
                    count++;
                }
            }
        }

        int localMean = sum / count;
        int threshold = localMean - constant;

        // Apply thresholding
        // if (_src[i * cols + j] >= threshold) {
        //     _dst[i * cols + j] = 0; // Foreground
        // } else {
        //     _dst[i * cols + j] = 255; // Background
        // }
        _dst[i * cols + j] = (_src[i * cols + j] >= threshold) ? 0 : 255;

    }
}

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

// __global__ void calculateMean(const unsigned char* _src, int* _mean, int rows, int cols, int blockSize) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;

//     __shared__ int sharedBlock[32][32];

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

//         _mean[i * cols + j] = sum / count;
//     }
// }

__global__ void calculateMean(const unsigned char* _src, int* _mean, int rows, int cols, int blockSize) {
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
        _mean[i * cols + j] = sum / count;
    }
}

// Function to calculate mean along rows
__global__ void calculateMeanRows(const unsigned char* _src, int* _mean, int rows, int cols, int blockSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int blockSize_1 = 13;
    int blockSize_2 = 23;
    int blockSize_3 = 33;


    if (i < rows && j < cols) {

        int sum = 0;
        int count = 0;

        // Calculate local mean along rows within the specified block size
        for (int x = -blockSize_1 / 2; x <= blockSize_1 / 2; x++) {
            int col = j + x;

            // Ensure the pixel is within bounds
            if (col >= 0 && col < cols) {
                sum += _src[i * cols + col];
                count++;
            }
        }
        _mean[i * cols + j] = sum / count;


        for (int x = -blockSize_2 / 2; x <= blockSize_2 / 2; x++) {

            if((x < -blockSize_1 / 2) || (x > blockSize_1 / 2)){
                int col = j + x;
                // Ensure the pixel is within bounds
                if (col >= 0 && col < cols) {
                    sum += _src[i * cols + col];
                    count++;
                }
            }
        }
        _mean[(i * cols + j) + rows * cols] = sum / count;

        for (int x = -blockSize_3 / 2; x <= blockSize_3 / 2; x++) {

            if((x < -blockSize_2 / 2) || (x > blockSize_2 / 2)){
                int col = j + x;
                // Ensure the pixel is within bounds
                if (col >= 0 && col < cols) {
                    sum += _src[i * cols + col];
                    count++;
                }
            }
        }
        _mean[(i * cols + j) + rows * cols * 2] = sum / count;
    }
}

// Function to calculate mean along columns
__global__ void calculateMeanCols(const int* _meanRows, int* _mean, int rows, int cols, int blockSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int blockSize_1 = 13;
    int blockSize_2 = 23;
    int blockSize_3 = 33;

    if (i < rows && j < cols) {

        int sum = 0;
        int count = 0;

        // Calculate local mean along columns within the specified block size
        for (int y = -blockSize_1 / 2; y <= blockSize_1 / 2; y++) {
            int row = i + y;

            // Ensure the pixel is within bounds
            if (row >= 0 && row < rows) {
                sum += _meanRows[row * cols + j];
                count++;
            }
        }
        _mean[i * cols + j] = sum / count;


        for (int y = -blockSize_2 / 2; y <= blockSize_2 / 2; y++) {

            if((y < -blockSize_1 / 2) || (y > blockSize_1 / 2)){
                int row = i + y;
                // Ensure the pixel is within bounds
                if (row >= 0 && row < rows) {
                    sum += _meanRows[row * cols + j + rows * cols];
                    count++;
                }
            }
        }
        _mean[(i * cols + j) + rows * cols] = sum / count;


        for (int y = -blockSize_3 / 2; y <= blockSize_3 / 2; y++) {

            if((y < -blockSize_2 / 2) || (y > blockSize_2 / 2)){
                int row = i + y;
                // Ensure the pixel is within bounds
                if (row >= 0 && row < rows) {
                    sum += _meanRows[row * cols + j + rows * cols * 2];
                    count++;
                }
            }
        }
        _mean[(i * cols + j) + rows * cols * 2] = sum / count;
    }
}


// __global__ void calculateMean_n(const unsigned char* _src, int* _mean, int rows, int cols, int blockSize) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;

//     if (i < rows && j < cols) {
//         int blockSize[3] = {13, 23, 33};
//         int sum[3] = {0};

//         for (int b = 0; b < 3; ++b) {
//             for (int x = -blockSize[b] / 2; x <= blockSize[b] / 2; x++) {
//                 for (int y = -blockSize[b] / 2; y <= blockSize[b] / 2; y++) {
//                     int row = i + x;
//                     int col = j + y;

//                     if (row >= 0 && row < rows && col >= 0 && col < cols) {
//                         sum[b] += _src[row * cols + col];
//                     }
//                 }
//             }
//             _mean[(i * cols + j) + b * rows * cols] = sum[b] / (blockSize[b] * blockSize[b]);
//         }
//     }
// }

__global__ void calculateMean_n(const unsigned char* _src, int* _mean, int rows, int cols, int blockSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int blockSize_1 = 13;
    int blockSize_2 = 23;
    int blockSize_3 = 33;

    if (i < rows && j < cols) {

        int sum = 0;
        int count = 0;

        // Calculate local mean within the specified block size
        for (int x = -blockSize_1 / 2; x <= blockSize_1 / 2; x++) {
            for (int y = -blockSize_1 / 2; y <= blockSize_1 / 2; y++) {
                int row = i + x;
                int col = j + y;

                // Ensure the pixel is within bounds
                if (row >= 0 && row < rows && col >= 0 && col < cols) {
                    sum += _src[row * cols + col];
                    count++;
                }
            }
        }
        _mean[i * cols + j] = sum / count;



        for (int x = -blockSize_2 / 2; x <= blockSize_2 / 2; x++) {
            for (int y = -blockSize_2 / 2; y <= blockSize_2 / 2; y++) {

                if((x < -blockSize_1 / 2) || (x > blockSize_1 / 2)){
                    if((y < -blockSize_1 / 2) || (y > blockSize_1 / 2)){
                        
                        int row = i + x;
                        int col = j + y;

                        // Ensure the pixel is within bounds
                        if (row >= 0 && row < rows && col >= 0 && col < cols) {
                            sum += _src[row * cols + col];
                            count++;
                        }
                    }
                }
            }
        }
        _mean[(i * cols + j) + rows * cols] = sum / count;

        for (int x = -blockSize_3 / 2; x <= blockSize_3 / 2; x++) {
            for (int y = -blockSize_3 / 2; y <= blockSize_3 / 2; y++) {

                if((x < -blockSize_2 / 2) || (x > blockSize_2 / 2)){
                    if((y < -blockSize_2 / 2) || (y > blockSize_2 / 2)){
                        
                        int row = i + x;
                        int col = j + y;

                        // Ensure the pixel is within bounds
                        if (row >= 0 && row < rows && col >= 0 && col < cols) {
                            sum += _src[row * cols + col];
                            count++;
                        }
                    }
                }
            }
        }
        _mean[(i * cols + j) + rows * cols * 2] = sum / count;
    }
}

__global__ void applyThreshold(const unsigned char* _src, unsigned char* _dst, const int* _mean, int rows, int cols, double c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols) {
        int threshold = _mean[i * cols + j] - c;

        // Apply thresholding
        if (_src[i * cols + j] >= threshold) {
            _dst[i * cols + j] = 0; // Foreground
        } else {
            _dst[i * cols + j] = 255; // Background
        }

        // _dst[i * cols + j] = (_src[i * cols + j] >= threshold) ? 0 : 255;
    }
}

__global__ void applyThreshold_n(const unsigned char* _src, unsigned char* _dst, int* _mean, int rows, int cols, double c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols) {
        int threshold = _mean[i * cols + j] - c;
        _dst[i * cols + j] = (_src[i * cols + j] >= threshold) ? 0 : 255;

        unsigned char* _dst2 = _dst + (rows * cols * sizeof(unsigned char));
        threshold = _mean[(i * cols + j) + rows * cols] - c;
        _dst2[i * cols + j] = (_src[i * cols + j] >= threshold) ? 0 : 255;

        unsigned char* _dst3 = _dst2 + (rows * cols * sizeof(unsigned char));
        threshold = _mean[(i * cols + j) + rows * cols * 2] - c;
        _dst3[i * cols + j] = (_src[i * cols + j] >= threshold) ? 0 : 255;
    }
}

static void threshold_kernel_cpu(const unsigned char* _src, unsigned char* _dst,
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
    // cudaMalloc((void**)&d_dst, dataSize);
    cudaMalloc((void**)&d_dst, dataSize*3);//for testinggggggggggggggggggggggg!!!
}

void CudaProcessor::update_image_to_VRAM(unsigned char* _src, unsigned char* _dst, unsigned char* d_src, unsigned char* d_dst, size_t dataSize){
    cudaMemcpy(d_src, _src, dataSize, cudaMemcpyHostToDevice);

    // cudaMemcpy(d_dst, _dst, dataSize, cudaMemcpyHostToDevice);//
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

    // clock_t time_used;
	// clock_t start = clock();

    threshold_kernel_cpu(_src, _dst, rows, cols, winSize, constant);

    // time_used = clock() - start;
    // double time_in_ms = static_cast<double>(time_used) / CLOCKS_PER_SEC * 1000.0;
    // printf("time usedcuda_threshold: %d\n", (int)time_in_ms);
}

#else

static int iDivUp(int a, int b) { return (a%b != 0) ? (a/b + 1) : (a/b); }


// Implement the cuda_threshold method
void CudaProcessor::cuda_threshold(const unsigned char* d_src, unsigned char* d_dst, int rows, int cols, int step, int winSize, double constant) {
    // printf("cuda_threshold_GPU %d\n", winSize);

	dim3 blocks(iDivUp(rows, 16), iDivUp(cols, 16));
	dim3 threads(16, 16);


    // threshold_kernel<<<blocks, threads>>>(d_src, d_dst, rows, cols, winSize, constant);

    int* d_mean;
    size_t meanSize = rows * cols * sizeof(int);
    cudaMalloc((void**)&d_mean, meanSize);

    calculateMean<<<blocks, threads>>>(d_src, d_mean, rows, cols, winSize);

    applyThreshold<<<blocks, threads>>>(d_src, d_dst, d_mean, rows, cols, constant);

    cudaFree(d_mean);
}

void CudaProcessor::cuda_threshold_n(const unsigned char* d_src, unsigned char* d_dst, int rows, int cols, int step, int winSize, double constant, int nScale) {
    // printf("cuda_threshold_GPU_n %d\n", winSize);

	dim3 blocks(iDivUp(rows, 16), iDivUp(cols, 16));
	dim3 threads(16, 16);

    // // Calculate the addresses for d_dst1, d_dst2, and d_dst3
    size_t dataSize = cols * rows * sizeof(unsigned char);

    // threshold_kernel<<<blocks, threads>>>(d_src, d_dst, rows, cols, winSize, constant);
    // threshold_kernel<<<blocks, threads>>>(d_src, d_dst + dataSize, rows, cols, 23, constant);
    // threshold_kernel<<<blocks, threads>>>(d_src, d_dst + 2 * dataSize, rows, cols, 33, constant);

    int* d_mean;
    int* d_meanRows;
    size_t meanSize = rows * cols * sizeof(int);//TODO
    cudaMalloc((void**)&d_meanRows, meanSize * 3);
    cudaMalloc((void**)&d_mean, meanSize * 3);


    // calculateMean<<<blocks, threads>>>(d_src, d_mean, rows, cols, winSize);
    // calculateMean<<<blocks, threads>>>(d_src, d_mean, rows, cols, 23);
    // calculateMean<<<blocks, threads>>>(d_src, d_mean, rows, cols, 33);

    // calculateMean_n<<<blocks, threads>>>(d_src, d_mean, rows, cols, winSize);

    // Call kernel for mean along rows
    calculateMeanRows<<<blocks, threads>>>(d_src, d_meanRows, rows, cols, winSize);

    // Call kernel for mean along columns
    calculateMeanCols<<<blocks, threads>>>(d_meanRows, d_mean, rows, cols, winSize);

    applyThreshold_n<<<blocks, threads>>>(d_src, d_dst, d_mean, rows, cols, constant);

    cudaFree(d_mean);
    cudaFree(d_meanRows);

    cudaDeviceSynchronize(); // Wait for all kernels to finish

}

#endif
    


} // namespace cu_aruco