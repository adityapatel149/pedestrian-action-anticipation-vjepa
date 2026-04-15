#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pedestrian_tracker_cpp/detector_runner_base.hpp"

class TensorRTDetector : public DetectorRunnerBase {
public:
    explicit TensorRTDetector(const std::string& engine_path, int device_id = 0);
    ~TensorRTDetector();

    TensorRTDetector(const TensorRTDetector&) = delete;
    TensorRTDetector& operator=(const TensorRTDetector&) = delete;

    DetectorOutput infer(const cv::Mat& bgr) override;
    int inputWidth() const override { return input_width_; }
    int inputHeight() const override { return input_height_; }

private:
    class Logger : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, const char* msg) noexcept override;
    };

    struct TRTDestroy
    {
        template <typename T>
        void operator()(T* obj) const
        {
            delete obj;
        }
    };

    template <typename T>
    using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

    struct DeviceBuffer
    {
        void* ptr{ nullptr };
        size_t bytes{ 0 };

        DeviceBuffer() = default;
        ~DeviceBuffer();

        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        DeviceBuffer(DeviceBuffer&& other) noexcept;
        DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

        void allocate(size_t nbytes);
    };

    struct EngineContext
    {
        TRTUniquePtr<nvinfer1::ICudaEngine> engine;
        TRTUniquePtr<nvinfer1::IExecutionContext> context;
        std::vector<DeviceBuffer> device_buffers;
        int nb_bindings{ 0 };
    };

    static std::vector<char> readFile(const std::string& path);
    static void checkCuda(cudaError_t status, const char* what);
    static size_t volume(const nvinfer1::Dims& dims);
    static int findBindingIndex(const nvinfer1::ICudaEngine& engine, const std::string& name);

    EngineContext loadEngine(const std::string& engine_path);
    std::vector<float> preprocess(const cv::Mat& bgr) const;
    void resolveModelInputShape();

private:
    Logger logger_;
    TRTUniquePtr<nvinfer1::IRuntime> runtime_;
    EngineContext detector_;
    cudaStream_t stream_{ nullptr };
    int device_id_{ 0 };

    std::string input_name_;
    std::string output_name_;

    int input_width_{ 640 };
    int input_height_{ 640 };
};