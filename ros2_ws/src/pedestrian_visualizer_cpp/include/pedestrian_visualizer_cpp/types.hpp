#pragma once

#include <vector>
#include <optional>

struct VizTrack
{
	int track_id;
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
};

struct VizPrediction
{
	int track_id;
	float crossing_prob;
	float risk_score;
	std::optional<float> distance_m;
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


struct BevConfig
{
	int bev_size{ 700 };
	float max_range_m{ 30.0f };
	float bev_half_width_m{ 12.0f };
};