#include "pedestrian_visualizer_cpp/bev.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_map>

namespace pedestrian_visualizer_cpp
{
    static cv::Scalar risk_to_color(float risk)
    {
        risk = std::clamp(risk, 0.0f, 1.0f);
        int r = static_cast<int>(255.0f * risk);
        int g = static_cast<int>(255.0f * (1.0f - risk));
        return cv::Scalar(0, g, r);
    }

    static cv::Mat build_bev_background(const BevConfig& bev_config)
    {
        cv::Mat bev(bev_config.bev_size, bev_config.bev_size, CV_8UC3, cv::Scalar(22, 22, 22));

        int cx = bev_config.bev_size / 2;
        int y_base = bev_config.bev_size - 20;
        int bonnet_w = static_cast<int>(0.34f * bev_config.bev_size);
        int bonnet_h = static_cast<int>(0.08f * bev_config.bev_size);

        cv::Mat overlay = bev.clone();

        cv::ellipse(
            overlay,
            cv::Point(cx, y_base),
            cv::Size(bonnet_w / 2, bonnet_h),
            0, 180, 360,
            cv::Scalar(255, 255, 255),
            2,
            cv::LINE_AA);

        cv::ellipse(
            overlay,
            cv::Point(cx, y_base),
            cv::Size(bonnet_w / 2, bonnet_h),
            0, 180, 360,
            cv::Scalar(55, 55, 55),
            2,
            cv::LINE_AA);

        int inner_w = static_cast<int>(bonnet_w * 0.72f);
        int inner_h = static_cast<int>(bonnet_h * 0.58f);

        cv::ellipse(
            overlay,
            cv::Point(cx, y_base + 1),
            cv::Size(inner_w / 2, inner_h),
            0, 180, 360,
            cv::Scalar(115, 115, 115),
            1,
            cv::LINE_AA);

        cv::addWeighted(overlay, 0.92, bev, 0.08, 0.0, bev);
        return bev;
    }

    static cv::Mat make_camera_matrix(const CameraConfig& cam)
    {
        return (cv::Mat_<float>(3, 3) <<
            cam.fx, 0.0f, cam.cx,
            0.0f, cam.fy, cam.cy,
            0.0f, 0.0f, 1.0f);
    }

    static cv::Mat make_distortion_vector(const CameraConfig& cam)
    {
        if (cam.dist_coeffs.empty()) {
            return cv::Mat::zeros(4, 1, CV_32F);
        }

        cv::Mat D(static_cast<int>(cam.dist_coeffs.size()), 1, CV_32F);
        for (size_t i = 0; i < cam.dist_coeffs.size(); ++i) {
            D.at<float>(static_cast<int>(i), 0) = cam.dist_coeffs[i];
        }
        return D;
    }

    // Visualization-only placement helper.
    static bool compute_ground_point(
        const cv::Mat& frame_bgr,
        const VizTrack& tr,
        const CameraConfig& cam,
        const BevConfig& bev_cfg,
        float& x_m,
        float& z_m)
    {
        const int h = frame_bgr.rows;
        const int w = frame_bgr.cols;

        float foot_x_norm = std::clamp((tr.x1 + tr.x2) * 0.5f, 0.0f, 1.0f);
        float foot_x = foot_x_norm * static_cast<float>(w - 1);

        float box_h = std::max(0.0f, tr.y2 - tr.y1);
        float foot_y_norm = std::min(1.0f, tr.y2 + 0.08f * box_h);
        float foot_y = foot_y_norm * static_cast<float>(h - 1);

        const cv::Mat K = make_camera_matrix(cam);
        const cv::Mat D = make_distortion_vector(cam);

        std::vector<cv::Point2f> src{ cv::Point2f(foot_x, foot_y) };
        std::vector<cv::Point2f> undist;
        cv::undistortPoints(src, undist, K, D);

        if (undist.empty()) {
            return false;
        }

        float x_n = undist[0].x;
        float y_n = undist[0].y;

        float pitch = cam.cam_pitch_deg * static_cast<float>(M_PI) / 180.0f;
        float beta = std::atan(y_n);

        float denom = std::tan(pitch + beta);
        if (std::abs(denom) < 1e-6f) {
            return false;
        }

        z_m = cam.cam_height_m / denom;
        if (z_m <= 0.0f) {
            return false;
        }

        x_m = x_n * z_m;

        if (std::abs(x_m) > bev_cfg.bev_half_width_m ||
            z_m < 0.0f || z_m > bev_cfg.max_range_m) {
            return false;
        }

        z_m -= 3.0f;
        return true;
    }

    static cv::Point world_to_bev(
        float x_m,
        float z_m,
        const BevConfig& cfg)
    {
        float x_norm = (x_m + cfg.bev_half_width_m) / (2.0f * cfg.bev_half_width_m);
        float y_norm = 1.0f - (z_m / cfg.max_range_m);

        int px = std::clamp(static_cast<int>(x_norm * (cfg.bev_size - 1)), 12, cfg.bev_size - 12);
        int py = std::clamp(static_cast<int>(y_norm * (cfg.bev_size - 1)), 0, cfg.bev_size - 1);
        return cv::Point(px, py);
    }

    cv::Mat render_bev(
        const cv::Mat& frame_bgr,
        const std::vector<VizTrack>& tracks,
        const std::unordered_map<int, VizPrediction>& predictions,
        const CameraConfig& camera_config,
        const BevConfig& bev_config)
    {
        cv::Mat bev = build_bev_background(bev_config);

        static std::unordered_map<int, cv::Point2f> previous_positions;
        const float alpha = 0.1f;

        for (const auto& tr : tracks) {
            float x_m = 0.0f;
            float z_m = 0.0f;

            if (!compute_ground_point(frame_bgr, tr, camera_config, bev_config, x_m, z_m)) {
                continue;
            }

            cv::Point raw_p = world_to_bev(x_m, z_m, bev_config);
            cv::Point2f p(static_cast<float>(raw_p.x), static_cast<float>(raw_p.y));

            auto prev_it = previous_positions.find(tr.track_id);
            if (prev_it != previous_positions.end()) {
                p.x = alpha * p.x + (1.0f - alpha) * prev_it->second.x;
                p.y = alpha * p.y + (1.0f - alpha) * prev_it->second.y;
            }
            previous_positions[tr.track_id] = p;

            cv::Point draw_p(static_cast<int>(p.x), static_cast<int>(p.y));
            draw_p.y -= static_cast<int>((3.0f / bev_config.max_range_m) * (bev_config.bev_size - 1));
            draw_p.y = std::clamp(draw_p.y, 0, bev_config.bev_size - 1);

            cv::Scalar color(255, 255, 255);
            char label[128];

            auto it = predictions.find(tr.track_id);
            if (it == predictions.end()) {
                std::snprintf(label, sizeof(label), "ID=%d", tr.track_id);
            }
            else {
                const auto& pred = it->second;
                color = risk_to_color(pred.risk_score);

                if (pred.distance_m.has_value()) {
                    std::snprintf(label, sizeof(label), "%.1fm r=%.2f",
                        *pred.distance_m, pred.risk_score);
                }
                else {
                    std::snprintf(label, sizeof(label), "r=%.2f", pred.risk_score);
                }
            }

            cv::circle(bev, draw_p, 4, color, 2, cv::LINE_AA);
            cv::putText(
                bev,
                label,
                cv::Point(std::min(bev.cols - 150, draw_p.x + 12), std::max(18, draw_p.y - 8)),
                cv::FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv::LINE_AA);
        }

        return bev;
    }
}