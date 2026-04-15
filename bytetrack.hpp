#pragma once

#include <vector>
#include <utility>

struct DetectionRow
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
};

enum class TrackState
{
    Tracked = 0,
    Lost = 1,
    Removed = 2
};

struct Track
{
    int id = -1;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float score = 0.0f;

    int age = 0;
    int time_since_update = 0;
    int start_frame = 0;
    int frame_id = 0;

    float vx = 0.0f;
    float vy = 0.0f;

    bool is_activated = false;
    TrackState state = TrackState::Tracked;
};

class ByteTracker
{
public:
    ByteTracker(
        float track_high_thresh = 0.6f,
        float track_low_thresh = 0.1f,
        float new_track_thresh = 0.7f,
        float match_thresh = 0.3f,
        int max_time_lost = 30);

    std::vector<Track> update(const std::vector<DetectionRow>& detections);

private:
    float computeIoU(const DetectionRow& det, const Track& track) const;
    float computeIoU(const Track& a, const Track& b) const;
    float centerDistance(const DetectionRow& det, const Track& track) const;
    float trackSize(const Track& track) const;

    void activateTrack(Track& track, int frame_id);
    void reActivateTrack(Track& track, const DetectionRow& det, int frame_id);
    void updateTrack(Track& track, const DetectionRow& det, int frame_id);
    Track predictTrack(const Track& track) const;

    std::vector<std::pair<int, int>> greedyMatch(
        const std::vector<Track>& tracks,
        const std::vector<DetectionRow>& detections,
        float iou_threshold,
        float distance_factor,
        std::vector<int>& unmatched_tracks,
        std::vector<int>& unmatched_detections) const;

private:
    std::vector<Track> tracked_tracks_;
    std::vector<Track> lost_tracks_;
    std::vector<Track> removed_tracks_;

    int next_track_id_ = 0;
    int frame_id_ = 0;

    float track_high_thresh_;
    float track_low_thresh_;
    float new_track_thresh_;
    float match_thresh_;
    int max_time_lost_;
};
