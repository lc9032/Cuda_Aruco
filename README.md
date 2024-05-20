# CUDA Optimization for ArUCO Marker Detection

## Overview

This project aims to enhance the performance of ArUCO marker detection in OpenCV by leveraging CUDA parallel computing capabilities, particularly on Nvidia Jetson TX2 devices.
The goal is to accelerate the detection process, especially for large images, by offloading certain computations to the GPU.

## Features

  - Improved performance of ArUCO marker detection using CUDA.
  - Offloaded thresholding computations to the GPU for faster processing.

## Usage
### Prerequisites

  - OpenCV library installed (version X.X.X).
  - Nvidia CUDA Toolkit installed (version X.X).

### Building

To build the project, follow these steps:

  - Clone the repository:

        git clone https://github.com/lc9032/Cuda_Aruco.git

  - Navigate to the project directory:

        cd Cuda_Aruco

  - Compile the project:

        mkdir build
        cd build
        cmake ..
        make

  - Run the executable:

        ./cuda_aruco

## Performance

Performance improvements were observed on Nvidia Jetson TX2 devices, particularly for large image sizes.
Debug mode showed more pronounced differences between the original OpenCV implementation and the CUDA-accelerated version.
