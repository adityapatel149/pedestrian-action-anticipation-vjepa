#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "pedestrian_tracker_cpp/detector_runner_base.hpp"

class OnnxDetector : public DetectorRunnerBase
{
public:
  explicit OnnxDetector(const std::string& model_path);
  DetectorOutput infer(const cv::Mat& bgr) override;
  int inputWidth() const override { return input_width_; }
  int inputHeight() const override { return input_height_; }

private:
  std::vector<float> preprocess(const cv::Mat& bgr) const;

  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::Session session_;

  std::string input_name_;
  std::string output_name_;

  int input_width_{640};
  int input_height_{640};
};
