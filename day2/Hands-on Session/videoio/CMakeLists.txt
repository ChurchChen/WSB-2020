project(videoio)
cmake_minimum_required(VERSION 2.8)
find_package(OpenCV REQUIRED)

set(base_dir ${PROJECT_SOURCE_DIR})
set(src_dir ${base_dir})

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(videoio ${src_dir}/videoio.cpp)
target_link_libraries(videoio ${OpenCV_LIBS})

