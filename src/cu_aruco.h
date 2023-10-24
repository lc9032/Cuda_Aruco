#ifndef _CU_ARUCO_H_
#define _CU_ARUCO_H_

namespace cu_aruco {

class CudaProcessor {
public:
    CudaProcessor(); // Constructor declaration

    bool InitCUDA();
    // Declare the cuda_threshold method
    void cuda_threshold(const unsigned char* _src, unsigned char* _dst, int rows, int cols, int step, int winSize, double constant);
};

} // namespace cu_aruco

#endif  // _CU_ARUCO_H_