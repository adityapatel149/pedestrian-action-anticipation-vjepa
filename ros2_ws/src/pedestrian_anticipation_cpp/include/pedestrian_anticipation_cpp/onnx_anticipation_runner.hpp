#pragma once

#include <string>
#include <vector>
#include <utility>
#include <onnxruntime_cxx_api.h>

#include "anticipation_runner_base.hpp"

class OnnxAnticipationRunner : public AnticipationRunnerBase
{
public:
    OnnxAnticipationRunner(
        const std::string& encoder_model,
        const std::string& classifier_model,
        bool use_cuda = true);

    std::vector<std::pair<int, float>> predict(
        const std::vector<float>& clip_cthw,
        const std::vector<int64_t>& clip_shape,      // [1,3,T,H,W]
        const std::vector<int>& track_ids,
        const std::vector<float>& bbox_tensor,
        const std::vector<int64_t>& bbox_shape,      // [N,T,4]
        float anticipation_time_sec) override;

private:
    std::vector<float> softmax2(const std::vector<float>& logits) const;

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session encoder_sess_;
    Ort::Session classifier_sess_;
};
