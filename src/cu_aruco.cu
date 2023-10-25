#include "cu_aruco.h"
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <vector>

// #include <opencv2/opencv.hpp>
// #include <opencv2/aruco.hpp>
// #include <opencv2/core.hpp>
// #include <opencv2/imgproc.hpp>
// #include "apriltag_quad_thresh.hpp"
// #include "zarray.hpp"

namespace cu_aruco {

__global__ void myfirstkernal(void){
	printf("Hello World from GPU!\n");
}

__global__ void threshold_kernel(const unsigned char* _src, unsigned char* _dst,
                                 int rows, int cols, int _pitch,
                                 double _param) {
	// printf("threshold_kernel!!!!!!\n");
	// printf("rows = %d, cols = %d \n", rows, cols);
//   int y = blockDim.y * blockIdx.y + threadIdx.y;
//   int x = blockDim.x * blockIdx.x + threadIdx.x;

//   if (y < rows && x < cols) {
// #pragma unroll
//     for (int i = 0; i < 5; ++i)
//       _dst[i * _pitch * rows + y * _pitch + x] = 
// 	_src[y * _pitch + x] > _dst[i * _pitch * rows + y * _pitch + x] - _param
// 				  ? 0
// 				  : 255;
//   }


  	// int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // if (tid < rows * cols) {
    //     // Copy data from _src to _dst
    //     _dst[tid] = _src[tid] > _param ? 255 : 0;
    // }

	for (int i = 0; i < rows * cols; i++){
		// _dst[i] = _src[i];
		_dst[i] = _src[i] > _param ? 255 : 0;
	}

}

__global__ void thresholdkernel (const unsigned char* _src, unsigned char* _dst, double maxValue,
								 int method, int type, int blockSize, double delta){
	printf("thresholdkernel\n");
}

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

// void adaptiveThreshold( const unsigned char* _src, unsigned char* _dst, int rows, int cols, int step,
// 							double maxValue, int method, int type, int blockSize, double delta )
// {
//     // CV_INSTRUMENT_REGION();

//     // Mat src = _src.getMat();
//     // CV_Assert( src.type() == CV_8UC1 );
//     // CV_Assert( blockSize % 2 == 1 && blockSize > 1 );
//     // Size size = src.size();

//     // _dst.create( size, src.type() ); //original
// 	_dst.create( size, CV_8U );
//     Mat dst = _dst.getMat();

//     if( maxValue < 0 )
//     {
//         dst = Scalar(0);
//         return;
//     }

//     // CALL_HAL(adaptiveThreshold, cv_hal_adaptiveThreshold, src.data, src.step, dst.data, dst.step, src.cols, src.rows,
//     //          maxValue, method, type, blockSize, delta);

//     Mat mean;

//     if( src.data != dst.data )
//         mean = dst;

//     if (method == ADAPTIVE_THRESH_MEAN_C)
//         boxFilter( src, mean, src.type(), Size(blockSize, blockSize),
//                    Point(-1,-1), true, BORDER_REPLICATE|BORDER_ISOLATED );
//     else if (method == ADAPTIVE_THRESH_GAUSSIAN_C)
//     {
//         Mat srcfloat,meanfloat;
//         src.convertTo(srcfloat,CV_32F);
//         meanfloat=srcfloat;
//         GaussianBlur(srcfloat, meanfloat, Size(blockSize, blockSize), 0, 0, BORDER_REPLICATE|BORDER_ISOLATED);
//         meanfloat.convertTo(mean, src.type());
//     }
//     else
//         CV_Error( CV_StsBadFlag, "Unknown/unsupported adaptive threshold method" );

//     int i, j;
//     uchar imaxval = saturate_cast<uchar>(maxValue);
//     int idelta = type == THRESH_BINARY ? cvCeil(delta) : cvFloor(delta);
//     uchar tab[768];

//     if( type == CV_THRESH_BINARY )
//         for( i = 0; i < 768; i++ )
//             tab[i] = (uchar)(i - 255 > -idelta ? imaxval : 0);
//     else if( type == CV_THRESH_BINARY_INV )
//         for( i = 0; i < 768; i++ )
//             tab[i] = (uchar)(i - 255 <= -idelta ? imaxval : 0);
//     else
//         CV_Error( CV_StsBadFlag, "Unknown/unsupported threshold type" );

//     if( src.isContinuous() && mean.isContinuous() && dst.isContinuous() )
//     {
//         size.width *= size.height;
//         size.height = 1;
//     }

//     for( i = 0; i < size.height; i++ )
//     {
//         const uchar* sdata = src.ptr(i);
//         const uchar* mdata = mean.ptr(i);
//         uchar* ddata = dst.ptr(i);

//         for( j = 0; j < size.width; j++ )
//             ddata[j] = tab[sdata[j] - mdata[j] + 255];
//     }
// }

static int iDivUp(int a, int b) { return (a%b != 0) ? (a/b + 1) : (a/b); }

// Implement the cuda_threshold method
void CudaProcessor::cuda_threshold(const unsigned char* _src, unsigned char* _dst, int rows, int cols, int step, int winSize, double constant) {

	dim3 blocks(iDivUp(rows, 16), iDivUp(cols, 16));
	dim3 threads(16, 16);

    // printf("cuda_threshold\n");

    // myfirstkernal<<<1,1>>>();

	unsigned char* d_src;  // Device memory for _src
    unsigned char* d_dst;  // Device memory for _dst

	size_t dataSize = rows * cols * sizeof(unsigned char);

	cudaMalloc((void**)&d_src, dataSize);
    cudaMalloc((void**)&d_dst, dataSize);

	cudaMemcpy(d_src, _src, dataSize, cudaMemcpyHostToDevice);


	// adaptiveThreshold(_src, _dst, rows, cols, step, 255, 0, 1, winSize, constant);

	thresholdkernel<<<1,1>>>(_src, _dst, 255, 0, 1, winSize, constant);

	// threshold_kernel<<<blocks, threads>>>(d_src, d_dst, rows, cols, step, 1);
	threshold_kernel<<<1, 1>>>(d_src, d_dst, rows, cols, step, 1);


	cudaMemcpy(_dst, d_dst, dataSize, cudaMemcpyDeviceToHost);

}

} // namespace cu_aruco