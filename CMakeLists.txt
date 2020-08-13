cmake_minimum_required(VERSION 3.0.0)
project(CSC4002W_Project VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)

add_executable(CSC4002W_Project main.cpp Image.cpp Preprocessing.cpp Segmentation.cpp Watershed.cpp)

target_link_libraries(CSC4002W_Project ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
