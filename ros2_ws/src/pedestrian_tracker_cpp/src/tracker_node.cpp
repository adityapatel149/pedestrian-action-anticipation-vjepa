#include <memory>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <array>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <motcpp/trackers/bytetrack.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp"

#include "pedestrian_interfaces/msg/bounding_box2_d_normalized.hpp"
#include "pedestrian_interfaces/msg/tracked_pedestrian.hpp"
#include "pedestrian_interfaces/msg/tracked_pedestrian_array.hpp"
#include "pedestrian_tracker_cpp/detector_runner_factory.hpp"

using namespace std::chrono_literals;

namespace
{
constexpr int kDetectionCols = 6;   // [x1, y1, x2, y2, conf, cls]
constexpr int kTrackCols = 8;       // [x1, y1, x2, y2, track_id, conf, cls, det_index]
}

class TrackerNode : public rclcpp::Node
{
public:
    TrackerNode() : Node("tracker_node")
    {
        const std::string image_topic = this->declare_parameter<std::string>("topics.image_topic", "/camera/image_raw");
        const std::string tracks_topic = this->declare_parameter<std::string>("topics.tracks_topic", "/pedestrian/tracks");
        const std::string model_path = this->declare_parameter<std::string>("tracker.model_path", "");
        const float det_thresh = this->declare_parameter<float>("tracker.det_thresh", 0.25);         
        const int max_age = this->declare_parameter<int>("tracker.max_age", 30);
        const int max_obs = this->declare_parameter<int>("tracker.max_obs", 50);
        const int min_hits = this->declare_parameter<int>("tracker.min_hits", 1);
        const float iou_threshold = this->declare_parameter<float>("tracker.iou_threshold", 0.2);
        const bool per_class = this->declare_parameter<bool>("tracker.per_class", false);
        const int nr_classes = this->declare_parameter<int>("tracker.nr_classes", 1);
        const std::string asso_func = this->declare_parameter<std::string>("tracker.asso_func", "iou");
        const bool is_obb = this->declare_parameter<bool>("tracker.is_obb", false);
        min_conf_ = static_cast<float>(this->declare_parameter<float>("tracker.min_conf", 0.25));
        const float track_thresh = this->declare_parameter<float>("tracker.track_thresh", 0.25);
        const float match_thresh = this->declare_parameter<float>("tracker.match_thresh", 0.5);
        const int track_buffer = this->declare_parameter<int>("tracker.track_buffer", 30);
        const int frame_rate = this->declare_parameter<int>("tracker.frame_rate", 30);
        
        if (model_path.empty()) {
            throw std::runtime_error("Parameter 'tracker.model_path' is empty");
        }

        tracker_ = std::make_unique<motcpp::trackers::ByteTrack>(
            static_cast<float>(det_thresh),
            max_age,
            max_obs,
            min_hits,
            static_cast<float>(iou_threshold),
            per_class,
            nr_classes,
            asso_func,
            is_obb,
            min_conf_,
            static_cast<float>(track_thresh),
            static_cast<float>(match_thresh),
            track_buffer,
            frame_rate);

        detector_ = createDetectorRunner(model_path);

        publisher_ = this->create_publisher<pedestrian_interfaces::msg::TrackedPedestrianArray>(
            tracks_topic, rclcpp::SensorDataQoS());

        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            image_topic,
            rclcpp::SensorDataQoS(),
            std::bind(&TrackerNode::imageCallback, this, std::placeholders::_1));
    }

private:
  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    const cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
    
    const auto output = detector_->infer(frame);
    
    const Eigen::MatrixXf detections = decodeDetections(
        output,
        frame.cols,
        frame.rows,
        detector_->inputWidth(),
        detector_->inputHeight(),
        min_conf_);

    const Eigen::MatrixXf tracks = tracker_->update(detections, frame);

    pedestrian_interfaces::msg::TrackedPedestrianArray out;
    out.header = msg->header;

    for (int i = 0; i < tracks.rows(); ++i) {
      if (tracks.cols() < kTrackCols) {
          RCLCPP_WARN(
              this->get_logger(),
              "Tracker output has %ld columns; expected at least %d",
              static_cast<long>(tracks.cols()), kTrackCols);
        break;
      }

      pedestrian_interfaces::msg::BoundingBox2DNormalized bbox;
      bbox.x1 = std::clamp(tracks(i, 0) / static_cast<float>(frame.cols), 0.0f, 1.0f);
      bbox.y1 = std::clamp(tracks(i, 1) / static_cast<float>(frame.rows), 0.0f, 1.0f);
      bbox.x2 = std::clamp(tracks(i, 2) / static_cast<float>(frame.cols), 0.0f, 1.0f);
      bbox.y2 = std::clamp(tracks(i, 3) / static_cast<float>(frame.rows), 0.0f, 1.0f);

      if (bbox.x2 <= bbox.x1 || bbox.y2 <= bbox.y1) {
        continue;
      }

      pedestrian_interfaces::msg::TrackedPedestrian track_msg;
      track_msg.track_id = static_cast<int>(std::lround(tracks(i, 4)));
      track_msg.bbox = bbox;
      track_msg.score = tracks(i, 5);
      out.tracks.push_back(track_msg);
    }

    publisher_->publish(out);
    RCLCPP_INFO(this->get_logger(), "Published %zu tracks", out.tracks.size());
  }

  Eigen::MatrixXf decodeDetections(
    const DetectorOutput& output,
    int frame_width,
    int frame_height,
    int model_width,
    int model_height,
    float score_threshold = 0.1f) const
  {
    if (output.shape.size() != 3 || output.shape[2] != kDetectionCols) {
      RCLCPP_WARN(this->get_logger(), "Unexpected detector output shape");
      return Eigen::MatrixXf(0, kDetectionCols);
    }

    const int num_det = static_cast<int>(output.shape[1]);
    const float* data = output.data.data();

    const float scale_x = static_cast<float>(frame_width) / static_cast<float>(model_width);
    const float scale_y = static_cast<float>(frame_height) / static_cast<float>(model_height);

    std::vector<std::array<float, kDetectionCols>> rows;
    rows.reserve(static_cast<size_t>(num_det));

    for (int i = 0; i < num_det; ++i) {
      const int base = i * kDetectionCols;

      float x1 = data[base + 0];
      float y1 = data[base + 1];
      float x2 = data[base + 2];
      float y2 = data[base + 3];
      const float score = data[base + 4];
      const int class_id = static_cast<int>(std::lround(data[base + 5]));

      if (score < score_threshold) {
        continue;
      }

      if (class_id != 0) {
        continue;
      }

      x1 = std::clamp(x1 * scale_x, 0.0f, static_cast<float>(frame_width - 1));
      y1 = std::clamp(y1 * scale_y, 0.0f, static_cast<float>(frame_height - 1));
      x2 = std::clamp(x2 * scale_x, 0.0f, static_cast<float>(frame_width - 1));
      y2 = std::clamp(y2 * scale_y, 0.0f, static_cast<float>(frame_height - 1));

      if (x2 <= x1 || y2 <= y1) {
        continue;
      }

      rows.push_back({x1, y1, x2, y2, score, static_cast<float>(class_id)});
    }

    Eigen::MatrixXf detections(static_cast<int>(rows.size()), kDetectionCols);
    for (int r = 0; r < static_cast<int>(rows.size()); ++r) {
      for (int c = 0; c < kDetectionCols; ++c) {
        detections(r, c) = rows[static_cast<size_t>(r)][static_cast<size_t>(c)];
      }
    }

    return detections;
  }

  rclcpp::Publisher<pedestrian_interfaces::msg::TrackedPedestrianArray>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  std::unique_ptr<DetectorRunnerBase> detector_;
  std::unique_ptr<motcpp::trackers::ByteTrack> tracker_;
  float min_conf_{ 0.25f };
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TrackerNode>());
  rclcpp::shutdown();
  return 0;
}
