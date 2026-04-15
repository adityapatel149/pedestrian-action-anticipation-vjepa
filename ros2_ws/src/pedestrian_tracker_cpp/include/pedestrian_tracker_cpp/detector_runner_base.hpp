#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdint>

struct DetectorOutput
{
    std::vector<int64_t> shape;
    std::vector<float> data;
};

class DetectorRunnerBase
{
public:
    virtual ~DetectorRunnerBase() = default;

    virtual DetectorOutput infer(const cv::Mat& bgr) = 0;
    virtual int inputWidth() const = 0;
    virtual int inputHeight() const = 0;
};