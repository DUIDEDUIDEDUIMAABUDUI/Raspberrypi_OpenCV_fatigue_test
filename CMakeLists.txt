cmake_minimum_required(VERSION 3.6)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

project(CamViewer
    DESCRIPTION "Qt application with fatigue detection and libcamera input"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Winvalid-pch -Wnon-virtual-dtor -Wextra -Wno-unused-parameter")

# 自动处理 Qt 元对象（MOC/UIC/RCC）
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# 查找依赖
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Qt5 COMPONENTS Widgets REQUIRED)

# 查找 libcamera
pkg_check_modules(LIBCAMERA REQUIRED IMPORTED_TARGET libcamera)

# 打印信息（可选）
message(STATUS "libcamera found")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# 包含头文件路径
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/libcam2opencv
    ${LIBCAMERA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# 添加子目录：libcam2opencv 封装模块
add_subdirectory(libcam2opencv)

# 添加可执行文件
add_executable(qtviewer
  main.cpp
  window.cpp
  fatigue_detector.cpp     # 疲劳检测模块
)

# 链接库（注意：直接写 qwt-qt5 而不是通过 pkg-config）
target_link_libraries(qtviewer
  Qt5::Widgets
  qwt-qt5                      # ✅ 直接链接，系统已装 .so 文件
  ${OpenCV_LIBS}
  PkgConfig::LIBCAMERA
  cam2opencv                  # ✅ 你自定义的封装库
lapack         # ✅ 添加 LAPACK 库
  blas           # ✅ 添加 BLAS 库（LAPACK 依赖）
)

