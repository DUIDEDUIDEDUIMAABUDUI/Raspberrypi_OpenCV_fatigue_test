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

    // 初始化热度计
    thermo = new QwtThermo;
    thermo->setFillBrush(QBrush(Qt::red));
    thermo->setScale(0, 255);
    thermo->show();

    image = new QLabel;

    hLayout = new QHBoxLayout();
    hLayout->addWidget(thermo);
    hLayout->addWidget(image);
    setLayout(hLayout);

    // 启动摄像头
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

// 异步处理 updateImage 函数
void Window::updateImage(const cv::Mat &mat) {
    cv::Mat input = mat.clone();  // 克隆图像用于后台处理

    std::thread([this, input]() {
        cv::Mat output;
        bool drowsy = detector.detect(input, output);  // 后台执行检测

        // BGR 转 RGB
        QImage frame(input.data, input.cols, input.rows, input.step, QImage::Format_RGB888);

        // 切回主线程更新 Qt 界面
        QMetaObject::invokeMethod(this, [this, frame, drowsy]() {
            image->setPixmap(QPixmap::fromImage(frame));
            const int h = frame.height();
            const int w = frame.width();
            const QColor c = frame.pixelColor(w / 2, h / 2);
            thermo->setValue(c.lightness());
            update();
        });
    }).detach();  // 分离线程，避免阻塞主线程
}
