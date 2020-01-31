cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 11)
project(face-recognition)

set( MTCNN_SRCS_DIR ${PROJECT_SOURCE_DIR}/src )
FILE(GLOB_RECURSE SOURCE_FILES ${MTCNN_SRCS_DIR}/*.cpp)

#if(WIN32)
#    set(OpenCV_DIR "E:/OpenCV/build/opencv-4.1.2")
#endif()
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${MTCNN_SRCS_DIR}/mtcnn ${PROJECT_SOURCE_DIR}/src)

add_executable(fr ${SOURCE_FILES})
target_link_libraries(fr ${OpenCV_LIBS})
