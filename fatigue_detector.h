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
    
    /**
     * @brief 执行疲劳检测
     * @param frame 输入图像帧
     * @param output 输出图像帧（包含检测框、关键点、状态文字等）
     * @return 是否处于疲劳状态（true 表示疲劳）
     */
    bool detect(const cv::Mat& frame, cv::Mat& output);

private:
    // EAR 计算函数
    float eye_aspect_ratio(const std::vector<cv::Point2f>& eye);
    
    // 提取左右眼区域关键点
    std::vector<cv::Point2f> extract_eye(const dlib::full_object_detection& shape, bool left);

    // Dlib 模型
    dlib::shape_predictor predictor;
    dlib::frontal_face_detector face_detector;

    // 状态跟踪（闭眼时长判断用）
    std::chrono::high_resolution_clock::time_point lastBlinkStart;
    bool eyeClosed;
    double eyeClosedDuration;

    // 卡尔曼滤波器（用于追踪眼中心）
    cv::KalmanFilter KF;
    bool kalmanInitialized = false;

    // EAR 阈值
    const float EAR_THRESHOLD = 0.21f;
};

#endif // FATIGUE_DETECTOR_H
