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

void Window::updateImage(const cv::Mat &mat) {
    cv::Mat input = mat.clone();

    std::thread([this, input]() {
        cv::Mat output;
        bool drowsy = detector.detect(input, output);  // 检测结果保存在 output 中

        if (output.empty()) return;

        // ❌ 不再转换颜色通道
        QImage frame(output.data, output.cols, output.rows, output.step, QImage::Format_BGR888);

        // ✅ 注意：此时格式应为 BGR888 而不是 RGB888
        QMetaObject::invokeMethod(this, [this, frame, drowsy]() {
            image->setPixmap(QPixmap::fromImage(frame));

            const int h = frame.height();
            const int w = frame.width();
            const QColor c = frame.pixelColor(w / 2, h / 2);
            thermo->setValue(c.lightness());

            update();
        });
    }).detach();
}
