#pragma once

#include <memory>
#include <string>

#include "anticipation_runner_base.hpp"
#include "onnx_anticipation_runner.hpp"
#include "tensorrt_anticipation_runner.hpp"

inline std::unique_ptr<AnticipationRunnerBase> createRunner(
    const std::string& encoder_path,
    const std::string& classifier_path)
{
    auto ends_with = [](const std::string& s, const std::string& suffix) {
        return s.size() >= suffix.size() &&
            s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
        };

    // ---- TensorRT path ----
    if (ends_with(encoder_path, ".engine") &&
        ends_with(classifier_path, ".engine"))
    {
        return std::make_unique<TensorRTAnticipationRunner>(
            encoder_path,
            classifier_path
        );
    }

    // ---- ONNX path ----
    if (ends_with(encoder_path, ".onnx") &&
        ends_with(classifier_path, ".onnx"))
    {
        return std::make_unique<OnnxAnticipationRunner>(
            encoder_path,
            classifier_path,
            true   // CUDA flag
        );
    }

    throw std::runtime_error(
        "Model format mismatch: both models must be either .onnx or .engine");
}