#include "src/aruco.h"
#include <iostream> 
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/aruco.hpp>
#include <thread>

#define CAPTURE_FRAME 0
#define CAL_TIME_CONSUME 0

using namespace std;

int main() {
    aruco::Aruco aruco;  // Create an instance of the Aruco class

    // init Cuda
    aruco.initCuda();

    cv::Mat frame;

#if CAPTURE_FRAME
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    cap >> frame;

#else

    cv::Mat markerImage;
    markerImage = cv::imread("pics/123_123.png");
    frame = markerImage.clone();

#endif

    cv::namedWindow("Camera Feed", cv::WINDOW_NORMAL);

    //malloc VRAM here
    unsigned char* d_src;  // Device memory for _src
    unsigned char* d_dst;  // Device memory for _dst
    size_t dataSize = frame.cols * frame.rows;

    printf("datasize = %d",frame.cols * frame.rows);

    aruco.codaMalloc_space_for_image(d_src, d_dst,dataSize);

#if CAPTURE_FRAME
    while(true) {
        
        // Capture a frame from the camera
        cap >> frame;

        // Check if the frame was captured successfully
        if (frame.empty()) {
            std::cerr << "Error: Could not capture frame." << std::endl;
            break;
        }

        cv::flip(frame, frame, 1);
#else
        
       
#endif

        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners;
        cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);

#if CAL_TIME_CONSUME
        clock_t time_used;
	    clock_t start = clock();
#endif

        aruco.detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, cv::noArray(), d_src, d_dst);
        // cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, cv::noArray());

#if CAL_TIME_CONSUME
        time_used = clock() - start;
        double time_in_ms = static_cast<double>(time_used) / CLOCKS_PER_SEC * 1000.0;
        printf("time used: %d\n", (int)time_in_ms);
#endif

        cv::aruco::drawDetectedMarkers(frame, markerCorners, markerIds);

        // Display the frame in the window
        cv::imshow("Camera Feed", frame);
        cv::waitKey(0);


#if CAPTURE_FRAME
        //Check for user input to exit (e.g., press 'q' key to quit)
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
#endif

    //free up VRAM
    aruco.free_up_VRAM(d_src, d_dst);

    cv::destroyAllWindows();

    return 0;
}