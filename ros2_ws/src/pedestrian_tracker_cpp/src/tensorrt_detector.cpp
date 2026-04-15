#include "pedestrian_tracker_cpp/tensorrt_detector.hpp"


#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace nvinfer1;

namespace
{
    template <typename T>
    std::vector<int64_t> dimsToShape(const T& dims)
    {
        std::vector<int64_t> out;
        out.reserve(static_cast<size_t>(dims.nbDims));
        for (int i = 0; i < dims.nbDims; ++i) {
            out.push_back(static_cast<int64_t>(dims.d[i]));
        }
        return out;
    }
}  // namespace

void TensorRTDetector::Logger::log(Severity severity, const char* msg) noexcept
{
    if (severity <= Severity::kWARNING) {
        std::cerr << "[TRT] " << msg << std::endl;
    }
}

TensorRTDetector::DeviceBuffer::~DeviceBuffer()
{
    if (ptr) {
        cudaFree(ptr);
    }
}

TensorRTDetector::DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
{
    ptr = other.ptr;
    bytes = other.bytes;
    other.ptr = nullptr;
    other.bytes = 0;
}

TensorRTDetector::DeviceBuffer& TensorRTDetector::DeviceBuffer::operator=(DeviceBuffer&& other) noexcept
{
    if (this != &other) {
        if (ptr) {
            cudaFree(ptr);
        }
        ptr = other.ptr;
        bytes = other.bytes;
        other.ptr = nullptr;
        other.bytes = 0;
    }
    return *this;
}

void TensorRTDetector::DeviceBuffer::allocate(size_t nbytes)
{
    if (ptr && nbytes <= bytes) {
        return;
    }

    if (ptr) {
        cudaFree(ptr);
    }

    TensorRTDetector::checkCuda(cudaMalloc(&ptr, nbytes), "cudaMalloc");
    bytes = nbytes;
}

TensorRTDetector::TensorRTDetector(const std::string& engine_path, int device_id)
    : device_id_(device_id)
{
    checkCuda(cudaSetDevice(device_id_), "cudaSetDevice");
    checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");

    runtime_.reset(createInferRuntime(logger_));
    if (!runtime_) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    detector_ = loadEngine(engine_path);

    if (!detector_.engine || !detector_.context) {
        throw std::runtime_error("Failed to load TensorRT detector engine");
    }

    if (detector_.engine->getNbIOTensors() < 2) {
        throw std::runtime_error("Detector engine must expose at least 1 input and 1 output tensor");
    }

    for (int i = 0; i < detector_.engine->getNbIOTensors(); ++i) {
        const char* tensor_name = detector_.engine->getIOTensorName(i);
        const auto mode = detector_.engine->getTensorIOMode(tensor_name);

        if (mode == TensorIOMode::kINPUT && input_name_.empty()) {
            input_name_ = tensor_name;
        }
        else if (mode == TensorIOMode::kOUTPUT && output_name_.empty()) {
            output_name_ = tensor_name;
        }
    }

    if (input_name_.empty()) {
        throw std::runtime_error("Could not find detector input tensor");
    }
    if (output_name_.empty()) {
        throw std::runtime_error("Could not find detector output tensor");
    }

    resolveModelInputShape();
}

TensorRTDetector::~TensorRTDetector()
{
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

std::vector<char> TensorRTDetector::readFile(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open engine file: " + path);
    }

    file.seekg(0, std::ios::end);
    const size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::vector<char> data(size);
    file.read(data.data(), static_cast<std::streamsize>(size));
    if (!file) {
        throw std::runtime_error("Failed to read engine file: " + path);
    }

    return data;
}

void TensorRTDetector::checkCuda(cudaError_t status, const char* what)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

size_t TensorRTDetector::volume(const Dims& dims)
{
    if (dims.nbDims <= 0) {
        throw std::runtime_error("Invalid dims");
    }

    size_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error("Unresolved or invalid tensor dimension");
        }
        v *= static_cast<size_t>(dims.d[i]);
    }
    return v;
}

int TensorRTDetector::findBindingIndex(const ICudaEngine& engine, const std::string& name)
{
    for (int i = 0; i < engine.getNbIOTensors(); ++i) {
        if (name == engine.getIOTensorName(i)) {
            return i;
        }
    }
    throw std::runtime_error("Tensor not found: " + name);
}

TensorRTDetector::EngineContext TensorRTDetector::loadEngine(const std::string& engine_path)
{
    EngineContext ec;
    auto blob = readFile(engine_path);

    ec.engine.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!ec.engine) {
        throw std::runtime_error("deserializeCudaEngine failed");
    }

    ec.context.reset(ec.engine->createExecutionContext());
    if (!ec.context) {
        throw std::runtime_error("createExecutionContext failed");
    }

    ec.nb_bindings = ec.engine->getNbIOTensors();
    ec.device_buffers.resize(static_cast<size_t>(ec.nb_bindings));

    return ec;
}

void TensorRTDetector::resolveModelInputShape()
{
    auto& ctx = *detector_.context;

    Dims input_dims = detector_.engine->getTensorShape(input_name_.c_str());

    // Expect NCHW
    if (input_dims.nbDims != 4) {
        throw std::runtime_error("Expected detector input to be 4D NCHW");
    }

    // Resolve dynamic dims conservatively as 1x3x640x640 if needed.
    Dims resolved = input_dims;
    if (resolved.d[0] <= 0) resolved.d[0] = 1;
    if (resolved.d[1] <= 0) resolved.d[1] = 3;
    if (resolved.d[2] <= 0) resolved.d[2] = 640;
    if (resolved.d[3] <= 0) resolved.d[3] = 640;

    if (!ctx.setInputShape(input_name_.c_str(), resolved)) {
        throw std::runtime_error("Failed to set detector input shape");
    }

#if NV_TENSORRT_MAJOR >= 10
    const int32_t unresolved = ctx.inferShapes(0, nullptr);
    if (unresolved < 0) {
        throw std::runtime_error("inferShapes failed for detector");
    }
#endif

    input_height_ = resolved.d[2];
    input_width_ = resolved.d[3];
}

std::vector<float> TensorRTDetector::preprocess(const cv::Mat& bgr) const
{
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(input_width_, input_height_));

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<float> input_tensor(static_cast<size_t>(3 * input_height_ * input_width_));

    std::vector<cv::Mat> chw(3);
    for (int i = 0; i < 3; ++i) {
        chw[i] = cv::Mat(
            input_height_,
            input_width_,
            CV_32F,
            input_tensor.data() + static_cast<size_t>(i) * input_height_ * input_width_);
    }

    cv::split(rgb, chw);
    return input_tensor;
}

DetectorOutput TensorRTDetector::infer(const cv::Mat& bgr)
{
    auto& ctx = *detector_.context;

    const std::vector<float> input_tensor_values = preprocess(bgr);

    Dims input_dims{};
    input_dims.nbDims = 4;
    input_dims.d[0] = 1;
    input_dims.d[1] = 3;
    input_dims.d[2] = input_height_;
    input_dims.d[3] = input_width_;

    if (!ctx.setInputShape(input_name_.c_str(), input_dims)) {
        throw std::runtime_error("Failed to set detector input shape before inference");
    }

#if NV_TENSORRT_MAJOR >= 10
    const int32_t unresolved = ctx.inferShapes(0, nullptr);
    if (unresolved < 0) {
        throw std::runtime_error("inferShapes failed before detector inference");
    }
#endif

    const Dims output_dims = ctx.getTensorShape(output_name_.c_str());

    const size_t input_bytes = input_tensor_values.size() * sizeof(float);
    const size_t output_bytes = volume(output_dims) * sizeof(float);

    const int input_idx = findBindingIndex(*detector_.engine, input_name_);
    const int output_idx = findBindingIndex(*detector_.engine, output_name_);

    detector_.device_buffers[static_cast<size_t>(input_idx)].allocate(input_bytes);
    detector_.device_buffers[static_cast<size_t>(output_idx)].allocate(output_bytes);

    checkCuda(
        cudaMemcpyAsync(
            detector_.device_buffers[static_cast<size_t>(input_idx)].ptr,
            input_tensor_values.data(),
            input_bytes,
            cudaMemcpyHostToDevice,
            stream_),
        "cudaMemcpyAsync input");

    if (!ctx.setTensorAddress(
        input_name_.c_str(),
        detector_.device_buffers[static_cast<size_t>(input_idx)].ptr)) {
        throw std::runtime_error("Failed to set detector input tensor address");
    }

    if (!ctx.setTensorAddress(
        output_name_.c_str(),
        detector_.device_buffers[static_cast<size_t>(output_idx)].ptr)) {
        throw std::runtime_error("Failed to set detector output tensor address");
    }

    if (!ctx.enqueueV3(stream_)) {
        throw std::runtime_error("Detector enqueueV3 failed");
    }

    DetectorOutput out;
    out.shape = dimsToShape(output_dims);
    out.data.resize(volume(output_dims));

    checkCuda(
        cudaMemcpyAsync(
            out.data.data(),
            detector_.device_buffers[static_cast<size_t>(output_idx)].ptr,
            output_bytes,
            cudaMemcpyDeviceToHost,
            stream_),
        "cudaMemcpyAsync output");

    checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");

    return out;
}