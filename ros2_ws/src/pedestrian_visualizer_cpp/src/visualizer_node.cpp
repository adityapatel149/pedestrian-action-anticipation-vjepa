#include <memory>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "pedestrian_interfaces/msg/tracked_pedestrian_array.hpp"
#include "pedestrian_interfaces/msg/pedestrian_prediction_array.hpp"

#include "pedestrian_visualizer_cpp/types.hpp"
#include "pedestrian_visualizer_cpp/overlay.hpp"
#include "pedestrian_visualizer_cpp/bev.hpp"

class VisualizerNode : public rclcpp::Node
{
public:
    VisualizerNode() : Node("visualizer_node")
    {
        const std::string image_topic =
            this->declare_parameter<std::string>("topics.image_topic", "/camera/image_raw");

        const std::string tracks_topic =
            this->declare_parameter<std::string>("topics.tracks_topic", "/pedestrian/tracks");

        const std::string predictions_topic =
            this->declare_parameter<std::string>("topics.predictions_topic", "/pedestrian/predictions");

        const std::string debug_image_topic =
            this->declare_parameter<std::string>("topics.debug_image_topic", "/pedestrian/debug_image");

        const std::string bev_image_topic =
            this->declare_parameter<std::string>("topics.bev_image_topic", "/pedestrian/bev_image");

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

        bev_config_.bev_size =
            this->declare_parameter<int>("bev.bev_size", 700);

        bev_config_.max_range_m =
            this->declare_parameter<float>("bev.max_range_m", 30.0);

        bev_config_.bev_half_width_m =
            this->declare_parameter<float>("bev.half_width_m", 12.0);


        auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();

        overlay_pub_ = this->create_publisher<sensor_msgs::msg::Image>(debug_image_topic, qos);
        bev_pub_ = this->create_publisher<sensor_msgs::msg::Image>(bev_image_topic, qos);

        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            image_topic, qos,
            std::bind(&VisualizerNode::imageCallback, this, std::placeholders::_1));

        tracks_sub_ = this->create_subscription<pedestrian_interfaces::msg::TrackedPedestrianArray>(
            tracks_topic, qos,
            std::bind(&VisualizerNode::tracksCallback, this, std::placeholders::_1));

        preds_sub_ = this->create_subscription<pedestrian_interfaces::msg::PedestrianPredictionArray>(
            predictions_topic, qos,
            std::bind(&VisualizerNode::predsCallback, this, std::placeholders::_1));
        
        RCLCPP_INFO(this->get_logger(), "Visualizer node started");
    }

private:
    std::vector<VizTrack> convertTracks(
        const pedestrian_interfaces::msg::TrackedPedestrianArray::SharedPtr& msg)
    {
        std::vector<VizTrack> out;
        out.reserve(msg->tracks.size());
        for (const auto& tr : msg->tracks) {
            out.push_back({ tr.track_id, tr.bbox.x1, tr.bbox.y1, tr.bbox.x2, tr.bbox.y2, tr.score });
        }
        return out;
    }

    std::unordered_map<int, VizPrediction> convertPreds(
        const pedestrian_interfaces::msg::PedestrianPredictionArray::SharedPtr& msg)
    {
        std::unordered_map<int, VizPrediction> out;
        for (const auto& pred : msg->predictions) {
            VizPrediction vp;
            vp.track_id = pred.track_id;
            vp.crossing_prob = pred.crossing_prob;
            vp.risk_score = pred.risk_score;
            if (pred.has_distance) {
                vp.distance_m = pred.distance_m;
            }
            out[pred.track_id] = vp;
        }
        return out;
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        latest_image_ = msg;
    }

    void predsCallback(const pedestrian_interfaces::msg::PedestrianPredictionArray::SharedPtr msg)
    {
        latest_preds_ = msg;
    }

    void tracksCallback(const pedestrian_interfaces::msg::TrackedPedestrianArray::SharedPtr msg)
    {
        latest_tracks_ = msg;
        if (!latest_image_) {
            return;
        }


        RCLCPP_INFO(this->get_logger(),
            "tracks=%zu preds=%zu",
            msg->tracks.size(),
            latest_preds_ ? latest_preds_->predictions.size() : 0);


        cv::Mat frame;
        try {
            frame = cv_bridge::toCvCopy(latest_image_, "bgr8")->image;
        }
        catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge failed: %s", e.what());
            return;
        }

        auto tracks = convertTracks(msg);
        auto preds = latest_preds_ ? convertPreds(latest_preds_) : std::unordered_map<int, VizPrediction>{};

        cv::Mat overlay = pedestrian_visualizer_cpp::draw_overlay(frame, tracks, preds);
        cv::Mat bev = pedestrian_visualizer_cpp::render_bev(frame, tracks, preds, camera_config_, bev_config_);

        auto overlay_msg = cv_bridge::CvImage(latest_image_->header, "bgr8", overlay).toImageMsg();
        auto bev_msg = cv_bridge::CvImage(latest_image_->header, "bgr8", bev).toImageMsg();

        overlay_pub_->publish(*overlay_msg);
        bev_pub_->publish(*bev_msg);
    }

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr overlay_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr bev_pub_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<pedestrian_interfaces::msg::TrackedPedestrianArray>::SharedPtr tracks_sub_;
    rclcpp::Subscription<pedestrian_interfaces::msg::PedestrianPredictionArray>::SharedPtr preds_sub_;

    sensor_msgs::msg::Image::SharedPtr latest_image_;
    pedestrian_interfaces::msg::TrackedPedestrianArray::SharedPtr latest_tracks_;
    pedestrian_interfaces::msg::PedestrianPredictionArray::SharedPtr latest_preds_;

    CameraConfig camera_config_;
    BevConfig bev_config_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VisualizerNode>());
    rclcpp::shutdown();
    return 0;
}