#include "window.h"
#include "fatigue_detector.h"

#include <iostream>
#include <thread>
#include <QMetaObject>
#include <atomic>

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
    

    // UI 布局
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

// 异步图像处理 + UI 更新（加锁保护）
void Window::updateImage(const cv::Mat &mat) {
    static std::atomic<bool> busy = false;
    if (busy) return;
    busy = true;

    cv::Mat input = mat.clone();

    std::thread([this, input]() {
        cv::Mat output;
        bool drowsy = detector.detect(input, output);  // 输出图像 + 疲劳状态

        if (output.empty()) {
            busy = false;
            return;
        }

        // BGR888 → Qt 图像（拷贝避免线程释放内存）
        QImage frame(output.data, output.cols, output.rows, output.step, QImage::Format_BGR888);
        QImage safeFrame = frame.copy();

        QMetaObject::invokeMethod(this, [this, safeFrame, drowsy]() {
            image->setPixmap(QPixmap::fromImage(safeFrame));

            const int h = safeFrame.height();
            const int w = safeFrame.width();
            const QColor c = safeFrame.pixelColor(w / 2, h / 2);
            thermo->setValue(c.lightness());


            update();
            busy = false;
        });
    }).detach();
}
