
project(aruco)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


FIND_PACKAGE(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_60;-lineinfo;--compiler-options;-O2")
SET(CUDA_PROPAGATE_HOST_FLAGS ON)

set(cuda_src cu_aruco.cu)
cuda_add_library(cu_aruco ${cuda_src})

add_library(aruco STATIC
    aruco.cpp
    aruco.h
    cu_aruco.h
    
    apriltag_quad_thresh.cpp
    apriltag_quad_thresh.hpp
    # charuco.cpp
    # dictionary.cpp
    precomp.hpp
    predefined_dictionaries_apriltag.hpp
    predefined_dictionaries.hpp
    unionfind.hpp
    zarray.hpp
    zmaxheap.cpp
    zmaxheap.hpp

    # Include other ARUCO source files here
)

# Link cu_aruco library to aruco
target_link_libraries(aruco cu_aruco ${OpenCV_LIBS})

