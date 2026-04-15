#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "anticipation_runner_base.hpp"

class TensorRTAnticipationRunner : public AnticipationRunnerBase
{
public:
    TensorRTAnticipationRunner(
        const std::string& encoder_engine,
        const std::string& classifier_engine,
        int device_id = 0);

    ~TensorRTAnticipationRunner();

    TensorRTAnticipationRunner(const TensorRTAnticipationRunner&) = delete;
    TensorRTAnticipationRunner& operator=(const TensorRTAnticipationRunner&) = delete;

    std::vector<std::pair<int, float>> predict(
        const std::vector<float>& clip_cthw,
        const std::vector<int64_t>& clip_shape,
        const std::vector<int>& track_ids,
        const std::vector<float>& bbox_tensor,
        const std::vector<int64_t>& bbox_shape,
        float anticipation_time_sec) override;

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
        std::vector<void*> bindings;
        std::vector<DeviceBuffer> device_buffers;
        int nb_bindings{ 0 };
    };

    EngineContext loadEngine(const std::string& engine_path);
    static std::vector<char> readFile(const std::string& path);
    static size_t volume(const nvinfer1::Dims& dims);
    static size_t elementSize(nvinfer1::DataType type);
    static void checkCuda(cudaError_t status, const char* what);
    static int findBindingIndex(const nvinfer1::ICudaEngine& engine, const std::string& name);
    static std::vector<float> softmax2(const std::vector<float>& logits);

    std::vector<float> runEncoder(
        const std::vector<float>& clip_cthw,
        const std::vector<int64_t>& clip_shape,
        float anticipation_time_sec,
        std::vector<int64_t>& output_shape);

    std::vector<float> runClassifier(
        const std::vector<float>& features,
        const std::vector<int64_t>& feature_shape,
        const std::vector<float>& bbox_tensor,
        const std::vector<int64_t>& bbox_shape);

private:
    Logger logger_;
    TRTUniquePtr<nvinfer1::IRuntime> runtime_;
    EngineContext encoder_;
    EngineContext classifier_;
    cudaStream_t stream_{ nullptr };
    int device_id_{ 0 };

    const std::string enc_clip_name_ = "clip";
    const std::string enc_ant_name_ = "anticipation_times";
    const std::string enc_out_name_ = "features";

    const std::string cls_feat_name_ = "features";
    const std::string cls_bbox_name_ = "bboxes";

    const std::string cls_cross_name_ = "cross";
    const std::string cls_action_name_ = "action";
    const std::string cls_intersection_name_ = "intersection";
    const std::string cls_signalized_name_ = "signalized";
};