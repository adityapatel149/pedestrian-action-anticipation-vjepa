cmake_minimum_required(VERSION 3.8)
project(pedestrian_anticipation_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pedestrian_interfaces REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
enable_language(CUDA)

include_directories(
  include
  /opt/onnxruntime/include
  /usr/include/x86_64-linux-gnu
  /usr/local/cuda/include
)

link_directories(
  /opt/onnxruntime/lib
  /usr/lib/x86_64-linux-gnu
  /usr/local/cuda/lib64
)

add_library(onnx_anticipation_runner
  src/onnx_anticipation_runner.cpp
)

target_link_libraries(onnx_anticipation_runner
  /opt/onnxruntime/lib/libonnxruntime.so
)

add_library(tensorrt_anticipation_runner
  src/tensorrt_anticipation_runner.cpp
)

target_link_libraries(tensorrt_anticipation_runner
  nvinfer
  nvinfer_plugin
  nvonnxparser
  cudart
)

add_executable(anticipation_node
  src/anticipation_node.cpp
)

target_link_libraries(anticipation_node
  onnx_anticipation_runner
  tensorrt_anticipation_runner
  ${OpenCV_LIBS}
)

ament_target_dependencies(onnx_anticipation_runner rclcpp)

ament_target_dependencies(anticipation_node
  rclcpp
  sensor_msgs
  std_msgs
  pedestrian_interfaces
  message_filters
  cv_bridge
)

install(TARGETS
  onnx_anticipation_runner
  tensorrt_anticipation_runner
  anticipation_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()