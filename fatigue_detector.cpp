#include "fatigue_detector.h"
#include <iostream>
#include <chrono>
#include <map>

FatigueDetector::FatigueDetector() {
    try {
        dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;
    } catch (std::exception &e) {
        std::cerr << "Failed to load shape predictor: " << e.what() << std::endl;
    }

    if (!face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
        std::cerr << "Failed to load Haar cascade!" << std::endl;
    }

    eyeClosed = false;
    eyeClosedDuration = 0.0;

    KF.init(4, 2, 0);
    KF.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0,
                                                     0, 1, 0, 1,
                                                     0, 0, 1, 0,
                                                     0, 0, 0, 1);
    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-5));
    setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));
    setIdentity(KF.errorCovPost, cv::Scalar::all(1));
}

float FatigueDetector::eye_aspect_ratio(const std::vector<cv::Point2f>& eye) {
    float A = cv::norm(eye[1] - eye[5]);
    float B = cv::norm(eye[2] - eye[4]);
    float C = cv::norm(eye[0] - eye[3]);
    return (A + B) / (2.0f * C);
}

std::vector<cv::Point2f> FatigueDetector::extract_eye(const dlib::full_object_detection& shape, bool left) {
    std::vector<cv::Point2f> eye;
    int start = left ? 36 : 42;
    for (int i = 0; i < 6; ++i)
        eye.emplace_back(shape.part(start + i).x(), shape.part(start + i).y());
    return eye;
}

std::map<std::string, double> FatigueDetector::calculateEBBA(double ear, double eyeClosedDuration) {
    std::map<std::string, double> ebba{
        {"NORMAL", 0.0},
        {"MEDIUM", 0.0},
        {"FATIGUE", 0.0}
    };

    if (ear > EAR_WARNING_THRESHOLD) {
        ebba["NORMAL"] = 0.9;
        ebba["MEDIUM"] = 0.05;
        ebba["FATIGUE"] = 0.05;
    } else if (ear > EAR_THRESHOLD) {
        ebba["NORMAL"] = 0.05;
        ebba["MEDIUM"] = 0.9;
        ebba["FATIGUE"] = 0.05;
    } else {
        if (eyeClosedDuration > 1.5) {
            ebba["NORMAL"] = 0.05;
            ebba["MEDIUM"] = 0.05;
            ebba["FATIGUE"] = 0.9;
        } else {
            ebba["NORMAL"] = 0.2;
            ebba["MEDIUM"] = 0.6;
            ebba["FATIGUE"] = 0.2;
        }
    }

    double total = ebba["NORMAL"] + ebba["MEDIUM"] + ebba["FATIGUE"];
    for (auto& b : ebba) b.second /= total;
    return ebba;
}

bool FatigueDetector::detect(const cv::Mat& frame, cv::Mat& output) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    output = frame.clone();

    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(80, 80));
    if (faces.empty()) return false;

    // 最大人脸
    cv::Rect biggest;
    int maxArea = 0;
    for (const auto& face : faces) {
        int area = face.width * face.height;
        if (area > maxArea) {
            maxArea = area;
            biggest = face;
        }
    }

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
    cv::circle(output, filtered_center, 4, cv::Scalar(0, 255, 255), -1);

    float ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0f;

    for (const auto& pt : left_eye) cv::circle(output, pt, 2, cv::Scalar(0, 255, 0), -1);
    for (const auto& pt : right_eye) cv::circle(output, pt, 2, cv::Scalar(0, 255, 0), -1);

    // EAR 状态融合判断
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

    return false;
}
