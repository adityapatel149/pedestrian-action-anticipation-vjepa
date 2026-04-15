#pragma once

#include <memory>
#include <string>
#include <stdexcept>

#include "pedestrian_tracker_cpp/detector_runner_base.hpp"
#include "pedestrian_tracker_cpp/onnx_detector.hpp"
#include "pedestrian_tracker_cpp/tensorrt_detector.hpp"

inline std::unique_ptr<DetectorRunnerBase> createDetectorRunner(
    const std::string& model_path)
{
    auto ends_with = [](const std::string& s, const std::string& suffix) {
        return s.size() >= suffix.size() &&
            s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
        };

    if (ends_with(model_path, ".engine")) {
        return std::make_unique<TensorRTDetector>(model_path);
    }

    if (ends_with(model_path, ".onnx")) {
        return std::make_unique<OnnxDetector>(model_path);
    }

    throw std::runtime_error(
        "Unsupported detector model format. Expected .onnx or .engine");
}