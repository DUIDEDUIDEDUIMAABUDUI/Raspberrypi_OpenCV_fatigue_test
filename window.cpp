#include "window.h"
#include "fatigue_detector.h"

#include <iostream>
#include <thread>
#include <QMetaObject>

FatigueDetector detector;

Window::Window()
{
    myCallback.window = this;
    camera.registerCallback(&myCallback);

    // 设置热度计
    thermo = new QwtThermo;
    thermo->setFillBrush(QBrush(Qt::red));
    thermo->setScale(0, 255);
    thermo->show();

    image = new QLabel;

    // UI 布局
    hLayout = new QHBoxLayout();
    hLayout->addWidget(thermo);
    hLayout->addWidget(image);
    setLayout(hLayout);

    // 启动摄像头采集
    Libcam2OpenCVSettings settings;
    settings.width = 800;
    settings.height = 600;
    settings.framerate = 30;
    camera.start(settings);
}

Window::~Window()
{
    camera.stop();
}

// 异步检测图像 + 颜色通道修复
void Window::updateImage(const cv::Mat &mat) {
    cv::Mat input = mat.clone();

    std::thread([this, input]() {
        cv::Mat output;
        bool drowsy = detector.detect(input, output);  // 包括人脸、EAR 等可视元素

        // ✅ BGR → RGB 颜色转换（防止颜色错乱）
        cv::Mat rgb;
        cv::cvtColor(output, rgb, cv::COLOR_BGR2RGB);

        // ✅ 构造 Qt 图像格式
        QImage frame(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);

        // ✅ 切回主线程更新 UI
        QMetaObject::invokeMethod(this, [this, frame, drowsy]() {
            image->setPixmap(QPixmap::fromImage(frame));

            const int h = frame.height();
            const int w = frame.width();
            const QColor c = frame.pixelColor(w / 2, h / 2);
            thermo->setValue(c.lightness());

            update();
        });
    }).detach();  // ✅ 分离线程，防止阻塞主线程
}
