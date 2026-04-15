#include <chrono>
#include <memory>
#include <functional>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include <future>
#include <atomic>
#include <cstring>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

#include "pedestrian_interfaces/msg/tracked_pedestrian_array.hpp"
#include "pedestrian_interfaces/msg/pedestrian_prediction.hpp"
#include "pedestrian_interfaces/msg/pedestrian_prediction_array.hpp"

#include "pedestrian_anticipation_cpp/runner_factory.hpp"

struct BBoxHistoryEntry
{
    float x1;
    float y1;
    float x2;
    float y2;
};

struct DetectionSnapshot
{
    int track_id;
    float x1, y1, x2, y2;
    float score;
};

struct CameraConfig
{
    float fx;
    float fy;
    float cx;
    float cy;
    float cam_height_m;
    float cam_pitch_deg;
    std::vector<float> dist_coeffs;
};

struct InferenceJob
{
    std::vector<float> clip;
    std::vector<int64_t> clip_shape;
    std::vector<int> track_ids;
    std::vector<float> bbox_tensor;
    std::vector<int64_t> bbox_shape;
    pedestrian_interfaces::msg::PedestrianPredictionArray out;
    std::vector<pedestrian_interfaces::msg::TrackedPedestrian> current_tracks;
};

class AnticipationNode : public rclcpp::Node
{
public:
    AnticipationNode() : Node("anticipation_node")
    {
        auto providers = Ort::GetAvailableProviders();
        for (auto& p : providers) {
            RCLCPP_INFO(this->get_logger(), "Provider: %s", p.c_str());
        }

        const std::string image_topic =
            this->declare_parameter<std::string>("topics.image_topic", "/camera/image_raw");

        const std::string tracks_topic =
            this->declare_parameter<std::string>("topics.tracks_topic", "/pedestrian/tracks");

        const std::string predictions_topic =
            this->declare_parameter<std::string>("topics.predictions_topic", "/pedestrian/predictions");

        encoder_model_ =
            this->declare_parameter<std::string>("anticipation.encoder_model", "");

        classifier_model_ =
            this->declare_parameter<std::string>("anticipation.classifier_model", "");

        input_fps_ =
            this->declare_parameter<float>("anticipation.input_fps", 30.0);

        model_fps_ =
            this->declare_parameter<float>("anticipation.model_fps", 15.0);

        clip_len_ =
            static_cast<size_t>(this->declare_parameter<int>("anticipation.clip_len", 8));

        resolution_ =
            this->declare_parameter<int>("anticipation.resolution", 256);

        max_boxes_ =
            static_cast<size_t>(this->declare_parameter<int>("anticipation.max_boxes", 8));

        infer_every_n_frames_ =
            this->declare_parameter<int>("anticipation.infer_every_n_frames", 6);

        camera_config_.fx =
            this->declare_parameter<float>("camera.fx", 1004.8374);

        camera_config_.fy =
            this->declare_parameter<float>("camera.fy", 1004.3913);

        camera_config_.cx =
            this->declare_parameter<float>("camera.cx", 960.10254);

        camera_config_.cy =
            this->declare_parameter<float>("camera.cy", 573.55383);

        camera_config_.cam_height_m =
            this->declare_parameter<float>("camera.cam_height_m", 1.27);

        camera_config_.cam_pitch_deg =
            this->declare_parameter<float>("camera.cam_pitch_deg", -10.0);

        const auto dist_coeffs_double =
            this->declare_parameter<std::vector<double>>(
                "camera.dist_coeffs",
                std::vector<double>{-0.027480543, -0.007055051, -0.039625194, 0.019310795});

        camera_config_.dist_coeffs.assign(dist_coeffs_double.begin(), dist_coeffs_double.end());

        max_range_m_ =
            this->declare_parameter<float>("bev.max_range_m", 30.0);

        bev_half_width_m_ =
            this->declare_parameter<float>("bev.half_width_m", 12.0);

        frame_width_ =
            this->declare_parameter<int>("bev.frame_width", 1920);

        frame_height_ =
            this->declare_parameter<int>("bev.frame_height", 1080);




        auto qos = rclcpp::QoS(rclcpp::KeepLast(10))
            .best_effort()
            .durability_volatile();

        predictions_pub_ = this->create_publisher<pedestrian_interfaces::msg::PedestrianPredictionArray>(
            predictions_topic, qos);

        image_sub_.subscribe(this, image_topic, rclcpp::SensorDataQoS().get_rmw_qos_profile());
        tracks_sub_.subscribe(this, tracks_topic, rclcpp::SensorDataQoS().get_rmw_qos_profile());

        Policy policy(10);
        policy.setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.1));
        sync_ = std::make_shared<Synchronizer>(policy);
        sync_->connectInput(image_sub_, tracks_sub_);

        sync_->registerCallback(
            std::bind(&AnticipationNode::syncCallback, this,
                std::placeholders::_1,
                std::placeholders::_2));

        frame_stride_ = static_cast<size_t>(std::round(input_fps_ / model_fps_));
        if (frame_stride_ < 1) {
            frame_stride_ = 1;
        }
        needed_ = 1 + (clip_len_ - 1) * frame_stride_;

        if (encoder_model_.empty() || classifier_model_.empty()) {
            throw std::runtime_error(
                "Parameters 'anticipation.encoder_model' or 'anticipation.classifier_model' are empty");
        }

        runner_ = createRunner(encoder_model_, classifier_model_);
        clip_shape_ = {
            1, 3,
            static_cast<int64_t>(clip_len_),
            static_cast<int64_t>(resolution_),
            static_cast<int64_t>(resolution_)
        };

        clip_buffer_.resize(3 * clip_len_ * resolution_ * resolution_);
        track_ids_buffer_.reserve(max_boxes_);
        bbox_tensor_buffer_.reserve(max_boxes_ * clip_len_ * 4);
        bbox_shape_buffer_.reserve(3);

        RCLCPP_INFO(this->get_logger(), "Anticipation node started");
    }

    ~AnticipationNode()
    {
        if (inference_future_.valid()) {
            inference_future_.wait();
        }
    }

private:
    using Policy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image,
        pedestrian_interfaces::msg::TrackedPedestrianArray>;

    using Synchronizer = message_filters::Synchronizer<Policy>;

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

    void syncCallback(
        const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
        const pedestrian_interfaces::msg::TrackedPedestrianArray::ConstSharedPtr& tracks_msg)
    {
        frame_index_++;
        updateFrameBuffer(image_msg);
        updateDetectionHistory(tracks_msg);

        if (frame_index_ % infer_every_n_frames_ != 0) {
            return;
        }

        pedestrian_interfaces::msg::PedestrianPredictionArray out;
        out.header = image_msg->header;

        if (frame_buffer_.size() < needed_ || detection_history_.size() < needed_) {
            predictions_pub_->publish(out);
            return;
        }

        if (!buildClipTensorCTHW(clip_buffer_)) {
            predictions_pub_->publish(out);
            return;
        }

        track_ids_buffer_.clear();
        bbox_tensor_buffer_.clear();
        bbox_shape_buffer_.clear();

        buildTrackSequences(track_ids_buffer_, bbox_tensor_buffer_, bbox_shape_buffer_);

        if (track_ids_buffer_.empty()) {
            predictions_pub_->publish(out);
            return;
        }

        if (inference_running_) {
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Running inference at frame %d", frame_index_);
        inference_running_ = true;

        InferenceJob job;
        job.clip = clip_buffer_;
        job.clip_shape = clip_shape_;
        job.track_ids = track_ids_buffer_;
        job.bbox_tensor = bbox_tensor_buffer_;
        job.bbox_shape = bbox_shape_buffer_;
        job.out = out;
        job.current_tracks.assign(tracks_msg->tracks.begin(), tracks_msg->tracks.end());

        inference_future_ = std::async(
            std::launch::async,
            [this, job = std::move(job)]() mutable
            {
                struct ResetFlag {
                    std::atomic<bool>& flag;
                    ~ResetFlag() { flag = false; }
                } reset{ inference_running_ };

                try {
                    auto t0 = std::chrono::steady_clock::now();

                    auto preds = runner_->predict(
                        job.clip,
                        job.clip_shape,
                        job.track_ids,
                        job.bbox_tensor,
                        job.bbox_shape,
                        anticipation_time_sec_);

                    auto t1 = std::chrono::steady_clock::now();
                    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

                    std::unordered_map<int, float> pred_map;
                    pred_map.reserve(preds.size());
                    for (const auto& p : preds) {
                        pred_map[p.first] = p.second;
                    }

                    auto result = job.out;

                    for (const auto& tr : job.current_tracks) {
                        auto it = pred_map.find(tr.track_id);
                        if (it == pred_map.end()) {
                            continue;
                        }

                        pedestrian_interfaces::msg::PedestrianPrediction pred;
                        pred.track_id = tr.track_id;
                        pred.bbox = tr.bbox;
                        pred.detection_score = tr.score;
                        pred.crossing_prob = it->second;

                        float x_m = 0.0f, z_m = 0.0f, dist_m = 0.0f;
                        if (computeGroundDistance(tr, frame_width_, frame_height_, x_m, z_m, dist_m)) {
                            pred.distance_m = dist_m;
                            pred.has_distance = true;
                            pred.risk_score = estimateRisk(pred.crossing_prob, dist_m, x_m);
                        }
                        else {
                            pred.distance_m = 0.0f;
                            pred.has_distance = false;
                            pred.risk_score = 0.0f;
                        }

                        result.predictions.push_back(pred);
                    }

                    predictions_pub_->publish(result);
                    RCLCPP_INFO(this->get_logger(), "Published %zu predictions in %ld ms",
                        result.predictions.size(), ms);
                }
                catch (const std::exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "Async anticipation inference failed: %s", e.what());
                }
            });
    }

    void updateFrameBuffer(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg)
    {
        frame_buffer_.push_back(image_msg);
        if (frame_buffer_.size() > max_frame_buffer_) {
            frame_buffer_.pop_front();
        }
    }

    std::vector<sensor_msgs::msg::Image::ConstSharedPtr> getRecentFrames()
    {
        std::vector<sensor_msgs::msg::Image::ConstSharedPtr> frames;

        if (frame_buffer_.size() < needed_) {
            return frames;
        }

        const size_t start_idx = frame_buffer_.size() - needed_;
        for (size_t i = 0; i < clip_len_; ++i) {
            size_t idx = start_idx + i * frame_stride_;
            frames.push_back(frame_buffer_[idx]);
        }

        return frames;
    }

    void updateDetectionHistory(const pedestrian_interfaces::msg::TrackedPedestrianArray::ConstSharedPtr& tracks_msg)
    {
        std::vector<DetectionSnapshot> dets;
        dets.reserve(tracks_msg->tracks.size());

        for (const auto& tr : tracks_msg->tracks) {
            dets.push_back({
                tr.track_id,
                tr.bbox.x1,
                tr.bbox.y1,
                tr.bbox.x2,
                tr.bbox.y2,
                tr.score
                });
        }

        detection_history_.push_back(dets);
        if (detection_history_.size() > needed_) {
            detection_history_.pop_front();
        }
    }

    void buildTrackSequences(
        std::vector<int>& track_ids,
        std::vector<float>& bbox_tensor,
        std::vector<int64_t>& bbox_shape)
    {
        track_ids.clear();
        bbox_tensor.clear();

        if (detection_history_.size() < needed_) {
            bbox_shape = { 0, 0, 0 };
            return;
        }

        const auto& current = detection_history_.back();
        if (current.empty()) {
            bbox_shape = { 0, 0, 0 };
            return;
        }

        struct MaybeBox {
            bool present{ false };
            BBoxHistoryEntry box{};
        };

        std::vector<DetectionSnapshot> current_sorted = current;
        std::sort(current_sorted.begin(), current_sorted.end(),
            [](const auto& a, const auto& b) { return a.score > b.score; });

        size_t kept = 0;
        for (const auto& det : current_sorted) {
            if (kept >= max_boxes_) {
                break;
            }

            std::vector<MaybeBox> seq(needed_);

            for (size_t t = 0; t < needed_; ++t) {
                for (const auto& cand : detection_history_[t]) {
                    if (cand.track_id == det.track_id) {
                        seq[t].present = true;
                        seq[t].box = { cand.x1, cand.y1, cand.x2, cand.y2 };
                        break;
                    }
                }
            }

            bool any = false;
            for (const auto& s : seq) {
                if (s.present) {
                    any = true;
                    break;
                }
            }
            if (!any) {
                continue;
            }

            size_t first = 0, last = 0;
            for (size_t i = 0; i < seq.size(); ++i) {
                if (seq[i].present) {
                    first = i;
                    break;
                }
            }
            for (size_t i = seq.size(); i-- > 0;) {
                if (seq[i].present) {
                    last = i;
                    break;
                }
            }

            for (size_t i = first; i-- > 0;) {
                seq[i].present = true;
                seq[i].box = seq[i + 1].box;
            }
            for (size_t i = last + 1; i < seq.size(); ++i) {
                seq[i].present = true;
                seq[i].box = seq[i - 1].box;
            }
            for (size_t i = first + 1; i < last; ++i) {
                if (!seq[i].present) {
                    seq[i].present = true;
                    seq[i].box = seq[i - 1].box;
                }
            }

            for (size_t i = 0; i < clip_len_; ++i) {
                size_t idx = i * frame_stride_;
                const auto& b = seq[idx].box;
                bbox_tensor.push_back(b.x1);
                bbox_tensor.push_back(b.y1);
                bbox_tensor.push_back(b.x2);
                bbox_tensor.push_back(b.y2);
            }

            track_ids.push_back(det.track_id);
            kept++;
        }

        bbox_shape = {
            static_cast<int64_t>(track_ids.size()),
            static_cast<int64_t>(clip_len_),
            4
        };
    }

    bool buildClipTensorCTHW(std::vector<float>& clip)
    {
        auto frames = getRecentFrames();
        if (frames.size() != clip_len_) {
            return false;
        }

        const size_t plane_size = static_cast<size_t>(resolution_) * resolution_;

        for (size_t t = 0; t < clip_len_; ++t) {
            cv::Mat frame;
            try {
                frame = cv_bridge::toCvCopy(frames[t], "bgr8")->image;
            }
            catch (const cv_bridge::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "cv_bridge failed: %s", e.what());
                return false;
            }

            cv::Mat rgb, resized, f32, normalized;
            cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
            cv::resize(rgb, resized, cv::Size(resolution_, resolution_), 0, 0, cv::INTER_LINEAR);
            resized.convertTo(f32, CV_32F, 1.0 / 255.0);

            cv::subtract(f32, cv::Scalar(mean_[0], mean_[1], mean_[2]), normalized);
            cv::divide(normalized, cv::Scalar(std_[0], std_[1], std_[2]), normalized);

            std::vector<cv::Mat> channels(3);
            cv::split(normalized, channels);

            for (int c = 0; c < 3; ++c) {
                if (!channels[c].isContinuous()) {
                    channels[c] = channels[c].clone();
                }

                float* dst = clip.data()
                    + static_cast<size_t>(c) * clip_len_ * plane_size
                    + static_cast<size_t>(t) * plane_size;

                std::memcpy(dst, channels[c].ptr<float>(), plane_size * sizeof(float));
            }
        }

        return true;
    }

    bool computeGroundDistance(
        const pedestrian_interfaces::msg::TrackedPedestrian& tr,
        int frame_w,
        int frame_h,
        float& x_m,
        float& z_m,
        float& dist_m) const
    {
        float foot_x_norm = std::clamp((tr.bbox.x1 + tr.bbox.x2) * 0.5f, 0.0f, 1.0f);
        float foot_x = foot_x_norm * static_cast<float>(frame_w - 1);

        float box_h = std::max(0.0f, tr.bbox.y2 - tr.bbox.y1);
        float foot_y_norm = std::min(1.0f, tr.bbox.y2 + 0.08f * box_h);
        float foot_y = foot_y_norm * static_cast<float>(frame_h - 1);

        const cv::Mat K = make_camera_matrix(camera_config_);
        const cv::Mat D = make_distortion_vector(camera_config_);

        std::vector<cv::Point2f> src{ cv::Point2f(foot_x, foot_y) };
        std::vector<cv::Point2f> undist;
        cv::undistortPoints(src, undist, K, D);

        if (undist.empty()) {
            return false;
        }

        float x_n = undist[0].x;
        float y_n = undist[0].y;

        float pitch = camera_config_.cam_pitch_deg * static_cast<float>(M_PI) / 180.0f;
        float beta = std::atan(y_n);

        float denom = std::tan(pitch + beta);
        if (std::abs(denom) < 1e-6f) {
            return false;
        }

        z_m = camera_config_.cam_height_m / denom;
        if (z_m <= 0.0f) {
            return false;
        }

        x_m = x_n * z_m;

        if (std::abs(x_m) > bev_half_width_m_ || z_m < 0.0f || z_m > max_range_m_) {
            return false;
        }

        z_m -= 3.0f;
        dist_m = std::sqrt(x_m * x_m + z_m * z_m);
        return true;
    }

    float estimateRisk(float cross_prob, float dist_m, float x_m) const
    {
        float dist_factor = 1.0f - std::clamp(dist_m / max_range_m_, 0.0f, 1.0f);
        float lateral_factor = 1.0f - std::clamp(std::abs(x_m) / bev_half_width_m_, 0.0f, 1.0f);
        return cross_prob * dist_factor * (0.5f + 0.5f * lateral_factor);
    }

    rclcpp::Publisher<pedestrian_interfaces::msg::PedestrianPredictionArray>::SharedPtr predictions_pub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
    message_filters::Subscriber<pedestrian_interfaces::msg::TrackedPedestrianArray> tracks_sub_;
    std::shared_ptr<Synchronizer> sync_;

    int frame_index_ = 0;

    std::deque<sensor_msgs::msg::Image::ConstSharedPtr> frame_buffer_;
    const size_t max_frame_buffer_ = 32;
    
    size_t frame_stride_ = 2;
    size_t needed_ = 8;

    const std::array<float, 3> mean_ = { 0.485f, 0.456f, 0.406f };
    const std::array<float, 3> std_ = { 0.229f, 0.224f, 0.225f };

    std::deque<std::vector<DetectionSnapshot>> detection_history_;

    std::string encoder_model_;
    std::string classifier_model_;
    std::unique_ptr<AnticipationRunnerBase> runner_;

    std::vector<float> clip_buffer_;
    
    const float anticipation_time_sec_ = 1.0f;
    size_t clip_len_ = 8;
    float input_fps_ = 30.0;
    float model_fps_ = 15.0;
    int resolution_ = 256;
    size_t max_boxes_ = 8;
    int infer_every_n_frames_ = 6;

    float max_range_m_ = 30.0;
    float bev_half_width_m_ = 12.0;
    int frame_width_ = 1920;
    int frame_height_ = 1080;

    std::vector<int64_t> clip_shape_{
        1, 3, static_cast<int64_t>(clip_len_),
        static_cast<int64_t>(resolution_),
        static_cast<int64_t>(resolution_)
    };

    std::vector<int> track_ids_buffer_;
    std::vector<float> bbox_tensor_buffer_;
    std::vector<int64_t> bbox_shape_buffer_;

    std::atomic<bool> inference_running_{ false };
    std::future<void> inference_future_;
   
    CameraConfig camera_config_{
        1004.8374f,
        1004.3913f,
        960.10254f,
        573.55383f,
        1.27f,
        -10.0f,
        {
            -0.027480543f,
            -0.007055051f,
            -0.039625194f,
            0.019310795f
        }
    };
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AnticipationNode>());
    rclcpp::shutdown();
    return 0;
}