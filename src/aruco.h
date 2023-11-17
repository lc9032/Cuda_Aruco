#ifndef _ARUCO_H_
#define _ARUCO_H_

#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>

#define CUDA_IMPLE 1 //0:OPENCV : 1:CUDA
#define SHOW_DEBUG_WINDOW 0 //0:OFF ; 1:ON
#define CPU_GPU_SWITCH 1 //0: CPU ; 1: GPU

namespace aruco {

using namespace cv;

class Aruco {
public:
    Aruco(); // Constructor declaration
    void initCuda(); // Member function declaration
    void detectMarkers(InputArray _image, const Ptr<cv::aruco::Dictionary> &_dictionary, OutputArrayOfArrays _corners,
                   OutputArray _ids, const Ptr<cv::aruco::DetectorParameters> &_params,
                   OutputArrayOfArrays _rejectedImgPoints);

};

} // namespace aruco

#endif  // _ARUCO_H_