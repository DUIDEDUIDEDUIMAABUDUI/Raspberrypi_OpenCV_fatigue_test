#include "fatigue_detector.h"
#include <iostream>
#include <chrono>
#include <map>

bool FatigueDetector::detect(const cv::Mat& frame, cv::Mat& output) {
    // ✅ 检查输入图像是否为彩色
    if (frame.channels() != 3) {
        std::cerr << "[ERROR] Input frame must be a 3-channel BGR color image!" << std::endl;
        return false;
    }

    output = frame.clone();  // ✅ 输出图像保留彩色（BGR）用于 Qt 显示
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);  // 灰度图用于人脸检测

    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(80, 80));
    if (faces.empty()) return false;

    // 找最大人脸
    cv::Rect biggest;
    int maxArea = 0;
    for (const auto& face : faces) {
        int area = face.width * face.height;
        if (area > maxArea) {
            maxArea = area;
            biggest = face;
        }
    }

    // 画人脸框
    cv::rectangle(output, biggest, cv::Scalar(255, 0, 0), 2);
    dlib::cv_image<dlib::bgr_pixel> cimg(frame);
    dlib::rectangle dlib_rect(biggest.x, biggest.y, biggest.x + biggest.width, biggest.y + biggest.height);
    dlib::full_object_detection shape = predictor(cimg, dlib_rect);

    auto left_eye = extract_eye(shape, true);
    auto right_eye = extract_eye(shape, false);

    // 卡尔曼滤波追踪眼角中心点
    cv::Point2f eye_center((left_eye[0].x + right_eye[3].x) / 2.0f,
                           (left_eye[0].y + right_eye[3].y) / 2.0f);
    cv::Mat_<float> measurement(2, 1);
    measurement(0) = eye_center.x;
    measurement(1) = eye_center.y;

    if (!kalmanInitialized) {
        KF.statePost.at<float>(0) = eye_center.x;
        KF.statePost.at<float>(1) = eye_center.y;
        KF.statePost.at<float>(2) = 0;
        KF.statePost.at<float>(3) = 0;
        kalmanInitialized = true;
    }

    cv::Mat prediction = KF.predict();
    cv::Mat estimated = KF.correct(measurement);
    cv::Point2f filtered_center(estimated.at<float>(0), estimated.at<float>(1));
    cv::circle(output, filtered_center, 4, cv::Scalar(0, 255, 255), -1);  // 显示稳定追踪点

    float ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0f;

    for (const auto& pt : left_eye)  cv::circle(output, pt, 2, cv::Scalar(0, 255, 0), -1);
    for (const auto& pt : right_eye) cv::circle(output, pt, 2, cv::Scalar(0, 255, 0), -1);

    // EAR 模糊判断逻辑
    auto now = std::chrono::high_resolution_clock::now();
    if (ear < EAR_THRESHOLD && !eyeClosed) {
        lastBlinkStart = now;
        eyeClosed = true;
    } else if (ear >= EAR_THRESHOLD && eyeClosed) {
        eyeClosedDuration = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastBlinkStart).count() / 1000.0;
        eyeClosed = false;
    }

    auto bba = calculateEBBA(ear, eyeClosedDuration);
    std::string label;
    double maxP = 0;
    for (const auto& [k, v] : bba) {
        if (v > maxP) {
            maxP = v;
            label = k;
        }
    }

    if (label == "FATIGUE") {
        cv::putText(output, "DROWSINESS ALERT!", cv::Point(50, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        return true;
    }

    // ✅ 输出图像通道数调试信息
    std::cerr << "[DEBUG] output.channels() = " << output.channels() << std::endl;

    return false;
}
