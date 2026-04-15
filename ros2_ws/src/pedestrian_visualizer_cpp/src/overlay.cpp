#include "pedestrian_visualizer_cpp/overlay.hpp"

#include <algorithm>
#include <cstdio>

namespace pedestrian_visualizer_cpp
{
    static cv::Scalar risk_to_color(float risk)
    {
        risk = std::clamp(risk, 0.0f, 1.0f);
        int r = static_cast<int>(255.0f * risk);
        int g = static_cast<int>(255.0f * (1.0f - risk));
        return cv::Scalar(0, g, r);
    }

    cv::Mat draw_overlay(
        const cv::Mat& frame_bgr,
        const std::vector<VizTrack>& tracks,
        const std::unordered_map<int, VizPrediction>& predictions)
    {
        cv::Mat canvas = frame_bgr.clone();
        const int h = canvas.rows;
        const int w = canvas.cols;

        for (const auto& tr : tracks) {
            int px1 = std::clamp(static_cast<int>(tr.x1 * w), 0, w - 1);
            int py1 = std::clamp(static_cast<int>(tr.y1 * h), 0, h - 1);
            int px2 = std::clamp(static_cast<int>(tr.x2 * w), 0, w - 1);
            int py2 = std::clamp(static_cast<int>(tr.y2 * h), 0, h - 1);

            auto it = predictions.find(tr.track_id);

            cv::Scalar color(255, 255, 255);
            char label[256];

            if (it == predictions.end()) {
                std::snprintf(label, sizeof(label), "ID=%d no-pred", tr.track_id);
            }
            else {
                const auto& pred = it->second;
                color = risk_to_color(pred.risk_score);
                if (pred.distance_m.has_value()) {
                    std::snprintf(
                        label,
                        sizeof(label),
                        "ID=%d P=%.2f Risk=%.2f %.1fm",
                        tr.track_id,
                        pred.crossing_prob,
                        pred.risk_score,
                        *pred.distance_m);
                }
                else {
                    std::snprintf(
                        label,
                        sizeof(label),
                        "ID=%d P=%.2f Risk=%.2f",
                        tr.track_id,
                        pred.crossing_prob,
                        pred.risk_score);
                }
            }

            cv::rectangle(canvas, cv::Point(px1, py1), cv::Point(px2, py2), color, 2);
            cv::rectangle(
                canvas,
                cv::Point(px1, std::max(0, py1 - 24)),
                cv::Point(std::min(w - 1, px1 + 260), py1),
                color,
                -1);

            cv::putText(
                canvas,
                label,
                cv::Point(px1 + 5, std::max(14, py1 - 7)),
                cv::FONT_HERSHEY_SIMPLEX,
                0.55,
                cv::Scalar(0, 0, 0),
                1,
                cv::LINE_AA);
        }

        return canvas;
    }
}