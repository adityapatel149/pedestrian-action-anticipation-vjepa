#pragma once

#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>

#include "pedestrian_visualizer_cpp/types.hpp"

namespace pedestrian_visualizer_cpp
{
	cv::Mat render_bev(
		const cv::Mat& frame_bgr,
		const std::vector<VizTrack>& tracks,
		const std::unordered_map<int, VizPrediction>& predictions,
		const CameraConfig& camera_config,
		const BevConfig& bev_config);
}