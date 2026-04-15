#include "pedestrian_tracker_cpp/onnx_detector.hpp"

#include <array>
#include <stdexcept>

OnnxDetector::OnnxDetector(const std::string& model_path)
: env_(ORT_LOGGING_LEVEL_WARNING, "pedestrian_tracker"),
  session_options_{},
  session_(nullptr)
{
  session_options_.SetIntraOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  OrtCUDAProviderOptions cuda_options{};
  cuda_options.device_id = 0;
  cuda_options.arena_extend_strategy = 0;
  cuda_options.gpu_mem_limit = SIZE_MAX;
  cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
  cuda_options.do_copy_in_default_stream = 1;

  session_options_.AppendExecutionProvider_CUDA(cuda_options);
  
  session_ = Ort::Session(env_, model_path.c_str(), session_options_);

  Ort::AllocatorWithDefaultOptions allocator;

  auto input_name_alloc = session_.GetInputNameAllocated(0, allocator);
  auto output_name_alloc = session_.GetOutputNameAllocated(0, allocator);

  input_name_ = input_name_alloc.get();
  output_name_ = output_name_alloc.get();
}


std::vector<float> OnnxDetector::preprocess(const cv::Mat& bgr) const
{
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(input_width_, input_height_));

  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

  rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

  std::vector<float> input_tensor(3 * input_height_ * input_width_);

  std::vector<cv::Mat> chw(3);
  for (int i = 0; i < 3; ++i) {
    chw[i] = cv::Mat(input_height_, input_width_, CV_32F,
                     input_tensor.data() + i * input_height_ * input_width_);
  }

  cv::split(rgb, chw);
  return input_tensor;
}

DetectorOutput OnnxDetector::infer(const cv::Mat& bgr)
{
  auto input_tensor_values = preprocess(bgr);

  std::array<int64_t, 4> input_shape{1, 3, input_height_, input_width_};
  const size_t input_tensor_size = input_tensor_values.size();

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
    OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info,
    input_tensor_values.data(),
    input_tensor_size,
    input_shape.data(),
    input_shape.size());

  const char* input_names[] = {input_name_.c_str()};
  const char* output_names[] = {output_name_.c_str()};

  auto output_tensors = session_.Run(
    Ort::RunOptions{nullptr},
    input_names,
    &input_tensor,
    1,
    output_names,
    1);

  float* output_data = output_tensors[0].GetTensorMutableData<float>();
  auto info = output_tensors[0].GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = info.GetShape();
  size_t count = info.GetElementCount();

  DetectorOutput out;
  out.shape = shape;
  out.data.assign(output_data, output_data + count);
  
  return out;
}
