#ifndef FATIGUE_DETECTOR_H
#define FATIGUE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>

class FatigueDetector {
public:
    FatigueDetector();

    // 检测主函数：输入原图，输出处理后图像（可绘图），返回是否疲劳
    bool detect(const cv::Mat& frame, cv::Mat& output);

private:
    // 模型
    dlib::shape_predictor predictor;
    cv::dnn::Net face_net;

    // 参数
    const float EAR_THRESHOLD = 0.25f;       // EAR判定阈值
    const int EYES_CLOSED_FRAMES = 15;       // 连续闭眼帧数
    int counter = 0;                         // 当前闭眼帧计数

    // 人脸追踪状态
    int frame_count = 0;                     // 当前帧计数
    bool has_face = false;                   // 当前是否有人脸可用
    cv::Rect last_face;                      // 最近一次人脸位置

    // 眼部关键点滤波
    bool landmark_initialized = false;
    std::vector<cv::Point2f> prev_left_eye;  // 上一帧左眼关键点
    std::vector<cv::Point2f> prev_right_eye; // 上一帧右眼关键点

    // 内部工具函数
    float eye_aspect_ratio(const std::vector<cv::Point2f>& eye);
    std::vector<cv::Point2f> extract_eye(const dlib::full_object_detection& shape, bool left);
};

#endif // FATIGUE_DETECTOR_H
