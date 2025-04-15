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

    
    thermo = new QwtThermo;
    thermo->setFillBrush(QBrush(Qt::red));
    thermo->setScale(0, 255);
    thermo->show();

    image = new QLabel;
    

    
    hLayout = new QHBoxLayout();
    hLayout->addWidget(thermo);
    hLayout->addWidget(image);
    
    setLayout(hLayout);

    
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
    static std::atomic<bool> busy = false;
    if (busy) return;
    busy = true;

    cv::Mat input = mat.clone();

    std::thread([this, input]() {
        cv::Mat output;
        bool drowsy = detector.detect(input, output);

        if (output.empty()) {
            busy = false;
            return;
        }

        QImage frame(output.data, output.cols, output.rows, output.step, QImage::Format_RGB888);
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
