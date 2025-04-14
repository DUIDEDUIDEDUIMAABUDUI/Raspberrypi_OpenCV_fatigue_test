#ifndef FATIGUE_DETECTOR_H
#define FATIGUE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <map>
#include <string>
#include <chrono>

class FatigueDetector {
public:
    FatigueDetector();
    bool detect(const cv::Mat& frame, cv::Mat& output);

private:
    
    float eye_aspect_ratio(const std::vector<cv::Point2f>& eye);
    std::vector<cv::Point2f> extract_eye(const dlib::full_object_detection& shape, bool left);

    
    std::map<std::string, double> calculateEBBA(double ear, double eyeClosedDuration);

    
    std::chrono::high_resolution_clock::time_point lastBlinkStart;
    bool eyeClosed;
    double eyeClosedDuration;

    
    dlib::shape_predictor predictor;
    dlib::frontal_face_detector face_detector;

    
    cv::KalmanFilter KF;
    bool kalmanInitialized = false;

    
    const float EAR_THRESHOLD = 0.21f;
    const float EAR_WARNING_THRESHOLD = 0.25f;
};

#endif
