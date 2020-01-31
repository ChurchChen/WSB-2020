cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 11)
project(object_detection)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(objdetector object_detection.cpp)
target_link_libraries(objdetector ${OpenCV_LIBS})
