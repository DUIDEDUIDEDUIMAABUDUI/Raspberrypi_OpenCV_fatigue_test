cmake_minimum_required(VERSION 3.6)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

project(CamViewer
    DESCRIPTION "Qt application with fatigue detection and libcamera input"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Winvalid-pch -Wnon-virtual-dtor -Wextra -Wno-unused-parameter")

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)


find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(dlib REQUIRED)


pkg_check_modules(LIBCAMERA REQUIRED IMPORTED_TARGET libcamera)


message(STATUS "libcamera found")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")


include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/libcam2opencv
    ${LIBCAMERA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)


add_subdirectory(libcam2opencv)


add_executable(qtviewer
  main.cpp
  window.cpp
  fatigue_detector.cpp     
)


target_link_libraries(qtviewer
  Qt5::Widgets
  qwt-qt5                     
  ${OpenCV_LIBS}
  PkgConfig::LIBCAMERA
  cam2opencv  
  dlib::dlib             
  lapack         
  blas          
)

