#include "fatigue_detector.h"
#include <iostream>

FatigueDetector::FatigueDetector() {
    try {
        dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;
        detector = dlib::get_frontal_face_detector();
    } catch (std::exception &e) {
        std::cerr << "Failed to load shape predictor: " << e.what() << std::endl;
    }

    KF.init(4, 2, 0);
    KF.transitionMatrix = (cv::Mat_<float>(4, 4) <<
                           1, 0, 1, 0,
                           0, 1, 0, 1,
                           0, 0, 1, 0,
                           0, 0, 0, 1);
    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-5));
    setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));
    setIdentity(KF.errorCovPost, cv::Scalar::all(1));
    measurement = cv::Mat_<float>(2, 1);
}

float FatigueDetector::eyeAspectRatio(const std::vector<dlib::point>& eye) {
    double A = dlib::length(eye[1] - eye[5]);
    double B = dlib::length(eye[2] - eye[4]);
    double C = dlib::length(eye[0] - eye[3]);
    return (A + B) / (2.0 * C);
}

float FatigueDetector::mouth_aspect_ratio(const std::vector<cv::Point>& mouth) {
    float A = cv::norm(mouth[2] - mouth[9]);
    float B = cv::norm(mouth[4] - mouth[7]);
    float C = cv::norm(mouth[0] - mouth[6]);
    return (A + B) / (2.0f * C);
}

std::map<std::string, double> FatigueDetector::calculateEBBA(double ear, double eyeClosedDuration) {
    std::map<std::string, double> ebba = {{"NORMAL", 0.0}, {"MEDIUM", 0.0}, {"FATIGUE", 0.0}};
    if (ear > EAR_WARNING_THRESHOLD) {
        ebba["NORMAL"] = 0.9;
    } else if (ear > EAR_DANGER_THRESHOLD) {
        ebba["MEDIUM"] = 0.9;
    } else {
        if (eyeClosedDuration > EYE_DURATION_THRESHOLD)
            ebba["FATIGUE"] = 0.98;
        else
            ebba["FATIGUE"] = 0.9;
    }
    double total = ebba["NORMAL"] + ebba["MEDIUM"] + ebba["FATIGUE"];
    for (auto &e : ebba) e.second /= total;
    return ebba;
}

std::map<std::string, double> FatigueDetector::calculateMBBA(double mar, double yawnDuration) {
    std::map<std::string, double> mbba = {{"Yawning", 0.0}, {"Speaking", 0.0}, {"Closing", 0.0}};
    if (mar < MAR_SPEAK_THRESHOLD) {
        mbba["Closing"] = 0.9;
    } else if (mar < MAR_YAWN_THRESHOLD) {
        mbba["Speaking"] = 0.9;
    } else {
        mbba["Yawning"] = yawnDuration > YAWN_DURATION_THRESHOLD ? 0.98 : 0.9;
    }
    double total = mbba["Yawning"] + mbba["Speaking"] + mbba["Closing"];
    for (auto &e : mbba) e.second /= total;
    return mbba;
}

std::map<std::string, double> FatigueDetector::combineBBA(const std::map<std::string, double>& bba1, const std::map<std::string, double>& bba2) {
    std::map<std::string, double> result;
    double total = 0;
    for (const auto &[k1, v1] : bba1) {
        for (const auto &[k2, v2] : bba2) {
            if (k1 == k2) {
                result[k1] += v1 * v2;
                total += v1 * v2;
            }
        }
    }
    for (auto &[k, v] : result) v /= total;
    return result;
}

bool FatigueDetector::detect(const cv::Mat& frame, cv::Mat& output) {
    output = frame.clone();
    dlib::cv_image<dlib::bgr_pixel> cimg(frame);
    std::vector<dlib::rectangle> faces = detector(cimg);
    if (faces.empty()) return false;

    dlib::full_object_detection shape = predictor(cimg, faces[0]);
    std::vector<dlib::point> left_eye, right_eye;
    for (int i = 36; i <= 41; ++i) left_eye.push_back(shape.part(i));
    for (int i = 42; i <= 47; ++i) right_eye.push_back(shape.part(i));
    float ear = (eyeAspectRatio(left_eye) + eyeAspectRatio(right_eye)) / 2.0f;

    auto now = std::chrono::high_resolution_clock::now();
    if (ear < EAR_DANGER_THRESHOLD && !eyeClosed) {
        lastBlinkStart = now;
        eyeClosed = true;
    } else if (ear >= EAR_DANGER_THRESHOLD && eyeClosed) {
        eyeClosedDuration = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastBlinkStart).count() / 1000.0;
        eyeClosed = false;
    }

    std::vector<cv::Point> mouth;
    for (int i = 48; i <= 59; ++i)
        mouth.emplace_back(shape.part(i).x(), shape.part(i).y());
    float mar = mouth_aspect_ratio(mouth);

    if (mar > MAR_YAWN_THRESHOLD && !yawnDetected) {
        lastYawnStart = now;
        yawnDetected = true;
    } else if (mar <= MAR_YAWN_THRESHOLD && yawnDetected) {
        yawnDuration = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastYawnStart).count() / 1000.0;
        yawnDetected = false;
    }

    auto ebba = calculateEBBA(ear, eyeClosedDuration);
    auto mbba = calculateMBBA(mar, yawnDuration);
    if (!prevEBBA.empty()) ebba = combineBBA(prevEBBA, ebba);
    if (!prevMBBA.empty()) mbba = combineBBA(prevMBBA, mbba);
    prevEBBA = ebba;
    prevMBBA = mbba;

    if (ebba["FATIGUE"] > HIGH_FATIGUE_THRESHOLD || mbba["Yawning"] > HIGH_FATIGUE_THRESHOLD) {
        cv::putText(output, "DROWSINESS ALERT!", cv::Point(50, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        return true;
    }

    cv::putText(output, "EAR: " + std::to_string(ear), cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1);
    cv::putText(output, "MAR: " + std::to_string(mar), cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 1);

    return false;
}
.5) {
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
