cmake_minimum_required(VERSION 3.8)
project(pedestrian_tracker_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

enable_language(CUDA)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pedestrian_interfaces REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(motcpp REQUIRED)
find_package(spdlog REQUIRED)

include_directories(
  include
  /opt/onnxruntime/include
  /usr/include/x86_64-linux-gnu
  /usr/local/cuda/include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

link_directories(
  /opt/onnxruntime/lib
  /usr/lib/x86_64-linux-gnu
  /usr/local/cuda/lib64
)

add_executable(tracker_node
  src/tracker_node.cpp
  src/onnx_detector.cpp
  src/tensorrt_detector.cpp
)

ament_target_dependencies(tracker_node
  rclcpp
  std_msgs
  sensor_msgs
  cv_bridge
  pedestrian_interfaces
)

target_link_libraries(tracker_node
  ${OpenCV_LIBS}
  /opt/onnxruntime/lib/libonnxruntime.so
  nvinfer
  nvinfer_plugin
  nvonnxparser
  cudart
  motcpp::motcpp
  spdlog::spdlog
)

install(TARGETS
  tracker_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()