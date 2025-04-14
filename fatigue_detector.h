#ifndef FATIGUE_DETECTOR_H
#define FATIGUE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <map>
#include <string>
#include <chrono>

class FatigueDetector {
public:
    FatigueDetector();
    bool detect(const cv::Mat& frame, cv::Mat& output);

private:
    // EAR 计算
    float eye_aspect_ratio(const std::vector<cv::Point2f>& eye);
    std::vector<cv::Point2f> extract_eye(const dlib::full_object_detection& shape, bool left);

    // BBA 模糊融合函数
    std::map<std::string, double> calculateEBBA(double ear, double eyeClosedDuration);

    // 闭眼判断状态
    std::chrono::high_resolution_clock::time_point lastBlinkStart;
    bool eyeClosed;
    double eyeClosedDuration;

    // 人脸检测和关键点
    dlib::shape_predictor predictor;
    cv::CascadeClassifier face_cascade;

    // 卡尔曼滤波器处理眼角稳定性
    cv::KalmanFilter KF;
    bool kalmanInitialized = false;

    // 阈值
    const float EAR_THRESHOLD = 0.21f;
    const float EAR_WARNING_THRESHOLD = 0.25f;
};

#endif
