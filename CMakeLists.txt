cmake_minimum_required(VERSION 3.10)
project(yolo_ncnn)

# Change this to your NCNN installation path
set(ncnn_DIR "/home/ncnn/build/install/lib/cmake/ncnn")
find_package(ncnn REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Add executables for YOLOv5 and YOLOv7 inference programs
add_executable(yolov5 yolov5.cpp)
target_link_libraries(yolov5 ncnn ${OpenCV_LIBS})

add_executable(yolov7 yolov7.cpp)
target_link_libraries(yolov7 ncnn ${OpenCV_LIBS})


