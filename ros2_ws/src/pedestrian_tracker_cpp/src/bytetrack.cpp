#include "pedestrian_tracker_cpp/bytetrack.hpp"

#include <algorithm>
#include <cmath>

ByteTracker::ByteTracker(
    float track_high_thresh,
    float track_low_thresh,
    float new_track_thresh,
    float match_thresh,
    int max_time_lost)
    : track_high_thresh_(track_high_thresh),
    track_low_thresh_(track_low_thresh),
    new_track_thresh_(new_track_thresh),
    match_thresh_(match_thresh),
    max_time_lost_(max_time_lost)
{}

float ByteTracker::computeIoU(const DetectionRow& det, const Track& track) const
{
    const float xx1 = std::max(det.x1, track.x1);
    const float yy1 = std::max(det.y1, track.y1);
    const float xx2 = std::min(det.x2, track.x2);
    const float yy2 = std::min(det.y2, track.y2);

    const float w = std::max(0.0f, xx2 - xx1);
    const float h = std::max(0.0f, yy2 - yy1);
    const float inter = w * h;

    const float area_det = (det.x2 - det.x1) * (det.y2 - det.y1);
    const float area_track = (track.x2 - track.x1) * (track.y2 - track.y1);
    const float union_area = area_det + area_track - inter;

    if (union_area <= 0.0f) {
        return 0.0f;
    }
    return inter / union_area;
}

float ByteTracker::computeIoU(const Track& a, const Track& b) const
{
    const float xx1 = std::max(a.x1, b.x1);
    const float yy1 = std::max(a.y1, b.y1);
    const float xx2 = std::min(a.x2, b.x2);
    const float yy2 = std::min(a.y2, b.y2);

    const float w = std::max(0.0f, xx2 - xx1);
    const float h = std::max(0.0f, yy2 - yy1);
    const float inter = w * h;

    const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    const float union_area = area_a + area_b - inter;

    if (union_area <= 0.0f) {
        return 0.0f;
    }
    return inter / union_area;
}

void ByteTracker::activateTrack(Track& track, int frame_id)
{
    track.id = next_track_id_++;
    track.age = 1;
    track.time_since_update = 0;
    track.start_frame = frame_id;
    track.frame_id = frame_id;
    track.state = TrackState::Tracked;
    track.is_activated = true;
}

void ByteTracker::reActivateTrack(Track& track, const DetectionRow& det, int frame_id)
{
    track.x1 = det.x1;
    track.y1 = det.y1;
    track.x2 = det.x2;
    track.y2 = det.y2;
    track.score = det.score;

    track.age += 1;
    track.time_since_update = 0;
    track.frame_id = frame_id;
    track.state = TrackState::Tracked;
    track.is_activated = true;
}

void ByteTracker::updateTrack(Track& track, const DetectionRow& det, int frame_id)
{
    track.x1 = det.x1;
    track.y1 = det.y1;
    track.x2 = det.x2;
    track.y2 = det.y2;
    track.score = det.score;

    track.age += 1;
    track.time_since_update = 0;
    track.frame_id = frame_id;
    track.state = TrackState::Tracked;
    track.is_activated = true;
}

std::vector<std::pair<int, int>> ByteTracker::greedyMatch(
    const std::vector<Track>& tracks,
    const std::vector<DetectionRow>& detections,
    float iou_threshold,
    std::vector<int>& unmatched_tracks,
    std::vector<int>& unmatched_detections) const
{
    struct Candidate
    {
        int t_idx;
        int d_idx;
        float iou;
    };

    std::vector<Candidate> candidates;

    for (size_t t = 0; t < tracks.size(); ++t) {
        for (size_t d = 0; d < detections.size(); ++d) {
            const float iou = computeIoU(detections[d], tracks[t]);
            if (iou >= iou_threshold) {
                candidates.push_back({ static_cast<int>(t), static_cast<int>(d), iou });
            }
        }
    }

    std::sort(
        candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) {
            return a.iou > b.iou;
        });

    std::vector<bool> track_used(tracks.size(), false);
    std::vector<bool> det_used(detections.size(), false);
    std::vector<std::pair<int, int>> matches;

    for (const auto& c : candidates) {
        if (track_used[c.t_idx] || det_used[c.d_idx]) {
            continue;
        }
        track_used[c.t_idx] = true;
        det_used[c.d_idx] = true;
        matches.emplace_back(c.t_idx, c.d_idx);
    }

    unmatched_tracks.clear();
    unmatched_detections.clear();

    for (size_t i = 0; i < tracks.size(); ++i) {
        if (!track_used[i]) {
            unmatched_tracks.push_back(static_cast<int>(i));
        }
    }

    for (size_t i = 0; i < detections.size(); ++i) {
        if (!det_used[i]) {
            unmatched_detections.push_back(static_cast<int>(i));
        }
    }

    return matches;
}

std::vector<Track> ByteTracker::update(const std::vector<DetectionRow>& detections)
{
    frame_id_++;

    std::vector<DetectionRow> high_dets;
    std::vector<DetectionRow> low_dets;

    for (const auto& det : detections) {
        if (det.score >= track_high_thresh_) {
            high_dets.push_back(det);
        }
        else if (det.score >= track_low_thresh_) {
            low_dets.push_back(det);
        }
    }

    std::vector<Track> track_pool = tracked_tracks_;
    track_pool.insert(track_pool.end(), lost_tracks_.begin(), lost_tracks_.end());

    std::vector<int> unmatched_pool;
    std::vector<int> unmatched_high;
    auto matches_high = greedyMatch(
        track_pool, high_dets, match_thresh_, unmatched_pool, unmatched_high);

    std::vector<Track> new_tracked;
    std::vector<Track> new_lost;
    std::vector<Track> new_removed;

    std::vector<bool> matched_tracked_old(tracked_tracks_.size(), false);
    std::vector<bool> matched_lost_old(lost_tracks_.size(), false);

    for (const auto& [pool_idx, det_idx] : matches_high) {
        Track track = track_pool[pool_idx];
        const auto& det = high_dets[det_idx];

        const bool from_tracked = pool_idx < static_cast<int>(tracked_tracks_.size());

        if (from_tracked) {
            matched_tracked_old[pool_idx] = true;
            updateTrack(track, det, frame_id_);
        }
        else {
            const int lost_idx = pool_idx - static_cast<int>(tracked_tracks_.size());
            matched_lost_old[lost_idx] = true;
            reActivateTrack(track, det, frame_id_);
        }

        new_tracked.push_back(track);
    }

    std::vector<Track> remain_tracked;
    for (size_t i = 0; i < tracked_tracks_.size(); ++i) {
        if (!matched_tracked_old[i]) {
            remain_tracked.push_back(tracked_tracks_[i]);
        }
    }

    std::vector<int> unmatched_remain_tracks;
    std::vector<int> unmatched_low;
    auto matches_low = greedyMatch(
        remain_tracked, low_dets, match_thresh_, unmatched_remain_tracks, unmatched_low);

    std::vector<bool> low_matched_remain(remain_tracked.size(), false);
    for (const auto& [trk_idx, det_idx] : matches_low) {
        Track track = remain_tracked[trk_idx];
        updateTrack(track, low_dets[det_idx], frame_id_);
        new_tracked.push_back(track);
        low_matched_remain[trk_idx] = true;
    }

    for (size_t i = 0; i < remain_tracked.size(); ++i) {
        if (!low_matched_remain[i]) {
            Track track = remain_tracked[i];
            track.time_since_update += 1;
            track.state = TrackState::Lost;
            new_lost.push_back(track);
        }
    }

    for (size_t i = 0; i < lost_tracks_.size(); ++i) {
        if (!matched_lost_old[i]) {
            Track track = lost_tracks_[i];
            track.time_since_update += 1;

            if (track.time_since_update > max_time_lost_) {
                track.state = TrackState::Removed;
                new_removed.push_back(track);
            }
            else {
                track.state = TrackState::Lost;
                new_lost.push_back(track);
            }
        }
    }

    for (int det_idx : unmatched_high) {
        const auto& det = high_dets[det_idx];
        if (det.score < new_track_thresh_) {
            continue;
        }

        Track track;
        track.x1 = det.x1;
        track.y1 = det.y1;
        track.x2 = det.x2;
        track.y2 = det.y2;
        track.score = det.score;
        activateTrack(track, frame_id_);
        new_tracked.push_back(track);
    }

    std::sort(
        new_tracked.begin(), new_tracked.end(),
        [](const Track& a, const Track& b) {
            return a.id < b.id;
        });

    new_tracked.erase(
        std::unique(
            new_tracked.begin(), new_tracked.end(),
            [](const Track& a, const Track& b) {
                return a.id == b.id;
            }),
        new_tracked.end());

    std::vector<Track> filtered_lost;
    for (const auto& lost : new_lost) {
        bool duplicate = false;
        for (const auto& tracked : new_tracked) {
            if (computeIoU(lost, tracked) > 0.85f) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            filtered_lost.push_back(lost);
        }
    }

    tracked_tracks_ = std::move(new_tracked);
    lost_tracks_ = std::move(filtered_lost);
    removed_tracks_.insert(removed_tracks_.end(), new_removed.begin(), new_removed.end());

    return tracked_tracks_;
}