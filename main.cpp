#include "window.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);  // 初始化 Qt 应用
    Window window;                 // 创建你的主窗口（含摄像头 + 疲劳检测）
    window.show();                 // 显示窗口
    return app.exec();             // 启动事件循环
}
