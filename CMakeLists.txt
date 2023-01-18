cmake_minimum_required(VERSION 3.21)
project(mnist_ncnn_executing)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(mnist_ncnn_executing main.cpp)

include_directories( ${CMAKE_CURRENT_LIST_DIR}/../ncnn/build-mac/src )
include_directories( ${CMAKE_CURRENT_LIST_DIR}/../ncnn/src )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

target_link_libraries( mnist_ncnn_executing ${OpenCV_LIBS} ${CMAKE_CURRENT_LIST_DIR}/../ncnn/build-mac/src/libncnn.a)