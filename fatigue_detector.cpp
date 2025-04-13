#include "fatigue_detector.h"
#include <iostream>

FatigueDetector::FatigueDetector() {
    try {
        dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;
    } catch (std::exception& e) {
        std::cerr << "无法加载关键点模型: " << e.what() << std::endl;
    }

    if (!face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
        std::cerr << "无法加载 Haar 模型" << std::endl;
    }
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

bool FatigueDetector::detect(const cv::Mat& frame, cv::Mat& output) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    output = frame.clone();

    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(80, 80));

    for (const auto& face : faces) {
        cv::rectangle(output, face, cv::Scalar(255, 0, 0), 2);

        dlib::cv_image<dlib::bgr_pixel> cimg(frame);
        dlib::rectangle dlib_rect(face.x, face.y, face.x + face.width, face.y + face.height);
        dlib::full_object_detection shape = predictor(cimg, dlib_rect);

        auto left_eye = extract_eye(shape, true);
        auto right_eye = extract_eye(shape, false);
        float ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0f;

        std::ostringstream oss;
        oss << "EAR: " << std::fixed << std::setprecision(2) << ear;
        cv::putText(output, oss.str(), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);

        for (const auto& pt : left_eye) cv::circle(output, pt, 2, cv::Scalar(0, 255, 0), -1);
        for (const auto& pt : right_eye) cv::circle(output, pt, 2, cv::Scalar(0, 255, 0), -1);

        if (ear < EAR_THRESHOLD) {
            counter++;
            if (counter >= EYES_CLOSED_FRAMES) {
                cv::putText(output, "DROWSINESS ALERT!", cv::Point(50, 70),
                            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
                return true;
            }
        } else {
            counter = 0;
        }
    }

    return false;
}
