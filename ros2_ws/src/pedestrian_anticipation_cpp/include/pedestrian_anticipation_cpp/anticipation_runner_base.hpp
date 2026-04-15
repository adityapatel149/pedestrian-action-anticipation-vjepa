#pragma once

#include <vector>
#include <utility>
#include <cstdint>

class AnticipationRunnerBase
{
public:
    virtual ~AnticipationRunnerBase() = default;

    virtual std::vector<std::pair<int, float>> predict(
        const std::vector<float>& clip_cthw,
        const std::vector<int64_t>& clip_shape,
        const std::vector<int>& track_ids,
        const std::vector<float>& bbox_tensor,
        const std::vector<int64_t>& bbox_shape,
        float anticipation_time_sec) = 0;
};