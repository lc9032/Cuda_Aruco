#ifndef _CU_ARUCO_H_
#define _CU_ARUCO_H_

#include "aruco.h"

namespace cu_aruco {

class CudaProcessor {
public:
    CudaProcessor(); // Constructor declaration

    bool InitCUDA();

    void codaMalloc_space_for_image(unsigned char*& d_src, unsigned char*& d_dst, size_t dataSize);

    void update_image_to_VRAM(unsigned char* _src, unsigned char* _dst, unsigned char* d_src, unsigned char* d_dst, size_t dataSize);

    void download_image_from_VRAM(unsigned char* _dst, unsigned char* d_dst, size_t dataSize);

    void free_up_VRAM(unsigned char* d_src, unsigned char* d_dst);

    void cuda_threshold(const unsigned char* _src, unsigned char* _dst, int rows, int cols, int step, int winSize, double constant);
};

} // namespace cu_aruco

#endif  // _CU_ARUCO_H_