cmake_minimum_required(VERSION 3.0)
project(CU_Aruco)


# Locate CUDA if you need it
# find_package(CUDA)
find_package(OpenCV REQUIRED)

# Add the src directory as a subdirectory
add_subdirectory(src)

# Link any necessary libraries and include directories
# target_link_libraries(YourExecutableOrLibrary ${OpenCV_LIBS} ...)

# If using CUDA, enable CUDA compilation for the cu_aruco.cu file
# if(CUDA_FOUND)
#     cuda_add_executable(YourCudaExecutable src/cu_aruco.cu)
# endif()

add_executable(main main.cpp)

target_link_libraries(main aruco ${OpenCV_LIBS})

