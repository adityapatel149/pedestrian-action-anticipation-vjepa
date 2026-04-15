#include "pedestrian_anticipation_cpp/onnx_anticipation_runner.hpp"
#include "rclcpp/rclcpp.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <iostream>


OnnxAnticipationRunner::OnnxAnticipationRunner(


    const std::string& encoder_model,
    const std::string& classifier_model,
    bool use_cuda)
    : env_(ORT_LOGGING_LEVEL_WARNING, "anticipation"),
    session_options_{},
    encoder_sess_(nullptr),
    classifier_sess_(nullptr)
{
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (use_cuda) {
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;

        session_options_.AppendExecutionProvider_CUDA(cuda_options);
    }

    encoder_sess_ = Ort::Session(env_, encoder_model.c_str(), session_options_);
    classifier_sess_ = Ort::Session(env_, classifier_model.c_str(), session_options_);
}

std::vector<float> OnnxAnticipationRunner::softmax2(const std::vector<float>& logits) const
{
    std::vector<float> out(logits.size());
    for (size_t i = 0; i < logits.size(); i += 2) {
        float a = logits[i];
        float b = logits[i + 1];
        float m = std::max(a, b);
        float ea = std::exp(a - m);
        float eb = std::exp(b - m);
        float s = ea + eb;
        out[i] = ea / s;
        out[i + 1] = eb / s;
    }
    return out;
}

std::vector<std::pair<int, float>> OnnxAnticipationRunner::predict(
    const std::vector<float>& clip_cthw,
    const std::vector<int64_t>& clip_shape,
    const std::vector<int>& track_ids,
    const std::vector<float>& bbox_tensor,
    const std::vector<int64_t>& bbox_shape,
    float anticipation_time_sec)
{
    std::vector<std::pair<int, float>> result;
    if (track_ids.empty()) {
        return result;
    }

    auto t0 = std::chrono::steady_clock::now();

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value clip = Ort::Value::CreateTensor<float>(
        mem,
        const_cast<float*>(clip_cthw.data()),
        clip_cthw.size(),
        clip_shape.data(),
        clip_shape.size());

    std::vector<float> ant_vec = { anticipation_time_sec };
    std::vector<int64_t> ant_shape = { 1 };
    Ort::Value ant = Ort::Value::CreateTensor<float>(
        mem,
        ant_vec.data(),
        ant_vec.size(),
        ant_shape.data(),
        ant_shape.size());

    const char* enc_in_names[] = { "clip", "anticipation_times" };
    const char* enc_out_names[] = { "features" };

    std::array<Ort::Value, 2> inputs{
        std::move(clip),
        std::move(ant)
    };

    auto t1 = std::chrono::steady_clock::now();
    auto enc_out = encoder_sess_.Run(
        Ort::RunOptions{ nullptr },
        enc_in_names,
        inputs.data(),
        2,
        enc_out_names,
        1);
    auto t2 = std::chrono::steady_clock::now();

    auto& feat_val = enc_out[0];
    auto feat_info = feat_val.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> feat_shape = feat_info.GetShape();
    size_t feat_count = feat_info.GetElementCount();
    float* feat_ptr = feat_val.GetTensorMutableData<float>();

    std::vector<int64_t> feat_batch_shape = feat_shape;
    feat_batch_shape[0] = static_cast<int64_t>(track_ids.size());

    size_t one_feat_count = feat_count;
    std::vector<float> feat_batch(one_feat_count * track_ids.size());

    for (size_t i = 0; i < track_ids.size(); ++i) {
        std::memcpy(
            feat_batch.data() + i * one_feat_count,
            feat_ptr,
            one_feat_count * sizeof(float));
    }

    auto t3 = std::chrono::steady_clock::now();

    Ort::Value features = Ort::Value::CreateTensor<float>(
        mem,
        feat_batch.data(),
        feat_batch.size(),
        feat_batch_shape.data(),
        feat_batch_shape.size());

    Ort::Value bboxes = Ort::Value::CreateTensor<float>(
        mem,
        const_cast<float*>(bbox_tensor.data()),
        bbox_tensor.size(),
        bbox_shape.data(),
        bbox_shape.size());

    const char* cls_in_names[] = { "features", "bboxes" };
    const char* cls_out_names[] = { "cross" };

    std::array<Ort::Value, 2> cls_inputs{
        std::move(features),
        std::move(bboxes)
    };

    auto cls_out = classifier_sess_.Run(
        Ort::RunOptions{ nullptr },
        cls_in_names,
        cls_inputs.data(),
        2,
        cls_out_names,
        1);
    auto t4 = std::chrono::steady_clock::now();

    auto& cross_val = cls_out[0];
    auto cross_info = cross_val.GetTensorTypeAndShapeInfo();
    size_t cross_count = cross_info.GetElementCount();
    float* cross_ptr = cross_val.GetTensorMutableData<float>();

    std::vector<float> logits(cross_ptr, cross_ptr + cross_count);
    auto probs = softmax2(logits);

    result.reserve(track_ids.size());
    for (size_t i = 0; i < track_ids.size(); ++i) {
        result.push_back({ track_ids[i], probs[i * 2 + 1] });
    }

    //auto prep_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    //auto enc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    //auto rep_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    //auto cls_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();

    //RCLCPP_INFO(
    //    rclcpp::get_logger("onnx_anticipation_runner"),
    //    "[runner] tracks=%zu prep_ms=%ld enc_ms=%ld repeat_ms=%ld cls_ms=%ld",
    //    track_ids.size(),
    //    prep_ms,
    //    enc_ms,
    //    rep_ms,
    //    cls_ms
    //);

    return result;
}