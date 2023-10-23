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

__global__ void thresholdkernel (){
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

// void adaptiveThreshold( InputArray _src, OutputArray _dst, double maxValue,
//                             int method, int type, int blockSize, double delta )
// {
//     CV_INSTRUMENT_REGION();

//     Mat src = _src.getMat();
//     CV_Assert( src.type() == CV_8UC1 );
//     CV_Assert( blockSize % 2 == 1 && blockSize > 1 );
//     Size size = src.size();

//     _dst.create( size, src.type() );
//     Mat dst = _dst.getMat();

//     if( maxValue < 0 )
//     {
//         dst = Scalar(0);
//         return;
//     }

//     CALL_HAL(adaptiveThreshold, cv_hal_adaptiveThreshold, src.data, src.step, dst.data, dst.step, src.cols, src.rows,
//              maxValue, method, type, blockSize, delta);

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

// Implement the cuda_threshold method
void CudaProcessor::cuda_threshold() {
    // printf("cuda_threshold\n");

    // myfirstkernal<<<1,1>>>();

	thresholdkernel<<<1,1>>>();


}

} // namespace cu_aruco