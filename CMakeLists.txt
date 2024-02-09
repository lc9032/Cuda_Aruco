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

add_executable(test_aruco test_aruco.cpp)

target_link_libraries(main aruco ${OpenCV_LIBS})

target_link_libraries(test_aruco aruco ${OpenCV_LIBS})



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")