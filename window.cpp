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

        cv::cvtColor(output, output, cv::COLOR_BGR2RGB);
        QImage frame(output.data, output.cols, output.rows, output.step, QImage::Format_RGB888);
        QImage safeFrame = frame.copy();

        QMetaObject::invokeMethod(this, [this, safeFrame, drowsy]() {
           
            image->setPixmap(QPixmap::fromImage(safeFrame));
            update();

            
            static int displayCount = 0;
            static auto lastDisplay = std::chrono::steady_clock::now();

            displayCount++;
            auto nowDisplay = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(nowDisplay - lastDisplay).count() >= 1) {
                qDebug() << "Display FPS:" << displayCount;
                displayCount = 0;
                lastDisplay = nowDisplay;
            }

            busy = false;
        });
    }).detach();
}

}
