#include "src/aruco.h"
#include <iostream> 
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio/videoio.hpp>
// #include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/aruco.hpp>

int main() {
    aruco::Aruco aruco;  // Create an instance of the Aruco class

    // Call the detect function
    aruco.initCuda();


    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    cv::namedWindow("Camera Feed", cv::WINDOW_NORMAL);

    // while (true) {
        cv::Mat frame;

        // Capture a frame from the camera
        cap >> frame;

        // Check if the frame was captured successfully
        // if (frame.empty()) {
        //     std::cerr << "Error: Could not capture frame." << std::endl;
        //     break;
        // }

        cv::flip(frame, frame, 1);

        //--------------------------------------------------
        cv::Mat markerImage;
        // markerImage = cv::imread("pics/123.jpg");
        markerImage = cv::imread("pics/123_123.png");
        frame = markerImage.clone();
        //--------------------------------------------------

        // std::vector<int> markerIds;
        // std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
        // cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
        // cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);
        // cv::aruco::ArucoDetector detector(dictionary, detectorParams);
        // detector.detectMarkers(frame, markerCorners, markerIds, rejectedCandidates);
        // cv::Mat outputImage = frame.clone();
        // cv::aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds);

        // std::vector<int> markerIds;
        // std::vector<std::vector<cv::Point2f>> markerCorners;
        // cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
        // cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);
        // cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, cv::noArray());
        // cv::aruco::drawDetectedMarkers(frame, markerCorners, markerIds);

        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners;
        cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);

        //---
        clock_t time_used;
	    clock_t start = clock();

        aruco.detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, cv::noArray());
        // cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, cv::noArray());

        time_used = clock() - start;
        printf("time used: %d\n", (int)time_used);
        //---

        cv::aruco::drawDetectedMarkers(frame, markerCorners, markerIds);

        // Display the frame in the window
        cv::imshow("Camera Feed", frame);
        cv::waitKey(0);

        // Check for user input to exit (e.g., press 'q' key to quit)
        // if (cv::waitKey(1) == 'q') {
        //     break;
        // }
    // }

    

    cap.release();
    cv::destroyAllWindows();

    return 0;
}