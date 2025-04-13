#include "window.h"
#include "fatigue_detector.h"

#include <iostream>  // ⬅️ 用于调试输出

FatigueDetector detector;

Window::Window()
{
    myCallback.window = this;
    camera.registerCallback(&myCallback);

    // UI 初始化
    thermo = new QwtThermo;
    thermo->setFillBrush(QBrush(Qt::red));
    thermo->setScale(0, 255);
    thermo->show();

    image = new QLabel;

    hLayout = new QHBoxLayout();
    hLayout->addWidget(thermo);
    hLayout->addWidget(image);
    setLayout(hLayout);

    // ✅ 正确调用 Libcam2OpenCV 封装的 start()，而不是直接 camera.start()
    Libcam2OpenCVSettings settings;
    settings.width = 800;
    settings.height = 600;
    settings.framerate = 30;
    camera.start(settings);  // ✅ 必须用封装的方法，才能初始化流和回调
}

Window::~Window()
{
    camera.stop();
}

void Window::updateImage(const cv::Mat &mat) {
    std::cerr << "[DEBUG] updateImage START" << std::endl;

    cv::Mat output;
    bool drowsy = detector.detect(mat, output);  // ⬅️ 疲劳检测处理图像

    cv::Mat rgb;
    cv::cvtColor(output, rgb, cv::COLOR_BGR2RGB);  // ⬅️ 转换为 RGB 格式，Qt 才能正确显示颜色

    const QImage frame(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    image->setPixmap(QPixmap::fromImage(frame));

    std::cerr << "[DEBUG] Drowsy = " << (drowsy ? "YES" : "NO") << std::endl;

    const int h = frame.height();
    const int w = frame.width();
    const QColor c = frame.pixelColor(w / 2, h / 2);
    thermo->setValue(c.lightness());

    update();
}
