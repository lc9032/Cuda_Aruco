#include "src/aruco.h"
#include <iostream> 
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/aruco.hpp>
#include <thread>


#define SHOW_DBG_WINDOWS 0

using namespace std;

int main() {
    aruco::Aruco aruco;  // Create an instance of the Aruco class

    // init Cuda
    aruco.initCuda();

    cv::Mat frame;

    cv::Mat markerImage;

/////////////////////////////////////////////////////////////////////////////////////
    // Define a vector to store paths to images
    std::vector<std::string> imagePaths;

    // Add paths to the vector
    imagePaths.push_back("pics/313x235.jpg");
    imagePaths.push_back("pics/626x470.jpg");
    imagePaths.push_back("pics/1252x939.jpg");
    imagePaths.push_back("pics/2504x1878.jpeg");
    imagePaths.push_back("pics/5008x3756.jpeg");


    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;

    std::vector<int> markerIds_CUDA;
    std::vector<std::vector<cv::Point2f>> markerCorners_CUDA;

    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);


    for (const auto& path : imagePaths) {
        std::cout << path << '\n';

        markerImage = cv::imread(path);
        frame = markerImage.clone();

        size_t dataSize = frame.cols * frame.rows * sizeof(unsigned char);
        int nScales = (parameters->adaptiveThreshWinSizeMax - parameters->adaptiveThreshWinSizeMin) / parameters->adaptiveThreshWinSizeStep + 1;
        aruco.codaMalloc_space_for_image(dataSize, dataSize * nScales);


        int numTrials = 10;
        double totalOpenCVTime = 0.0;
        double totalCUDATime = 0.0;

        for (int trial = 0; trial < numTrials; ++trial) {
            clock_t start;

            // OpenCV detection
            start = clock();
            cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, cv::noArray());
            totalOpenCVTime += static_cast<double>(clock() - start) / CLOCKS_PER_SEC * 1000.0;

            // CUDA detection
            start = clock();
            aruco.detectMarkers(frame, dictionary, markerCorners_CUDA, markerIds_CUDA, parameters, cv::noArray());
            totalCUDATime += static_cast<double>(clock() - start) / CLOCKS_PER_SEC * 1000.0;
        }

        // Print average times
        printf("Average time used (OpenCV): %.2f ms\n", totalOpenCVTime / numTrials);
        printf("Average time used (CUDA): %.2f ms\n", totalCUDATime / numTrials);


        //compare markerIds
        bool resultsMatch = true;
        for (size_t i = 0; i < markerCorners.size(); i++) {
            if (markerIds[i] != markerIds_CUDA[i]) {
                resultsMatch = false;
                printf("Marker ID mismatch at index %zu\n", i);
            }
        }
        if (resultsMatch) {
            printf("Marker detection results match between OpenCV and CUDA.\n");
        } else {
            printf("Marker detection results differ between OpenCV and CUDA.\n");
        }

#if SHOW_DBG_WINDOWS
        // Display the frame in the window
        cv::aruco::drawDetectedMarkers(frame, markerCorners_CUDA, markerIds_CUDA);
        cv::imshow("Camera Feed", frame);

        cv::waitKey(0);

        cv::destroyAllWindows();
#endif
        
        //free up VRAM
        aruco.free_up_VRAM();

        printf("-------------------------------------------------------\n");
    }
//////////////////////////////////////////////////////////////////////////////////////

/*

    
    markerImage = cv::imread("pics/test1.png");
    frame = markerImage.clone();






    cv::namedWindow("Camera Feed", cv::WINDOW_NORMAL);

    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);

    //malloc VRAM here
    size_t dataSize = frame.cols * frame.rows * sizeof(unsigned char);
    int nScales = (parameters->adaptiveThreshWinSizeMax - parameters->adaptiveThreshWinSizeMin) / parameters->adaptiveThreshWinSizeStep + 1;
    aruco.codaMalloc_space_for_image(dataSize, dataSize * nScales);

    clock_t time_used;
	clock_t start = clock();

    aruco.detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, cv::noArray());
    // cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, cv::noArray());

    time_used = clock() - start;
    double time_in_ms = static_cast<double>(time_used) / CLOCKS_PER_SEC * 1000.0;
    printf("time used: %d\n", (int)time_in_ms);

    cv::aruco::drawDetectedMarkers(frame, markerCorners, markerIds);

    // Display the frame in the window
    cv::imshow("Camera Feed", frame);

    cv::waitKey(0);

    //free up VRAM
    aruco.free_up_VRAM();

    cv::destroyAllWindows();
*/

    return 0;
}

