#ifndef _ARUCO_H_
#define _ARUCO_H_

#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>

#define CUDA_IMPLE 1 //0:OPENCV : 1:CUDA (for testing, will be removed)
#define SHOW_DEBUG_WINDOW 0 //0:OFF ; 1:ON
#define SHOW_DEBUG_MSG 0 //0:OFF ; 1:ON
#define CPU_GPU_SWITCH 1 //0: CPU ; 1: GPU (for testing, will be removed)
#define PARALLEL_FOR_IMPLE 2//0: _cudaThreshold, 1: _cudaThreshold_n, 2: Parallel (for testing, will be removed)

#define OPENCV_VER 0 //1: for Jetson

namespace aruco {

using namespace cv;

class Aruco {
public:
    Aruco(); // Constructor declaration
    void initCuda(); // Member function declaration

    void codaMalloc_space_for_image(unsigned char*& d_src, unsigned char*& d_dst, size_t dataSize);

    void free_up_VRAM(unsigned char* d_src, unsigned char* d_dst);

    void detectMarkers(InputArray _image, const Ptr<cv::aruco::Dictionary> &_dictionary, OutputArrayOfArrays _corners,
                   OutputArray _ids, const Ptr<cv::aruco::DetectorParameters> &_params,
                   OutputArrayOfArrays _rejectedImgPoints,
                   unsigned char* d_src, unsigned char* d_dst);

};

} // namespace aruco

#endif  // _ARUCO_H_