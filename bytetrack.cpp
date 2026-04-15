#include "pedestrian_tracker_cpp/bytetrack.hpp"

#include <algorithm>
#include <cmath>

namespace
{
float centerX(const DetectionRow& det)
{
    return 0.5f * (det.x1 + det.x2);
}

float centerY(const DetectionRow& det)
{
    return 0.5f * (det.y1 + det.y2);
}

float centerX(const Track& track)
{
    return 0.5f * (track.x1 + track.x2);
}

float centerY(const Track& track)
{
    return 0.5f * (track.y1 + track.y2);
}
} // namespace

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

float ByteTracker::centerDistance(const DetectionRow& det, const Track& track) const
{
    const float dx = centerX(det) - centerX(track);
    const float dy = centerY(det) - centerY(track);
    return std::sqrt(dx * dx + dy * dy);
}

float ByteTracker::trackSize(const Track& track) const
{
    const float width = std::max(0.0f, track.x2 - track.x1);
    const float height = std::max(0.0f, track.y2 - track.y1);
    return std::max(1.0f, std::sqrt(width * width + height * height));
}

Track ByteTracker::predictTrack(const Track& track) const
{
    Track predicted = track;
    predicted.x1 += track.vx;
    predicted.x2 += track.vx;
    predicted.y1 += track.vy;
    predicted.y2 += track.vy;
    return predicted;
}

void ByteTracker::activateTrack(Track& track, int frame_id)
{
    track.id = next_track_id_++;
    track.age = 1;
    track.time_since_update = 0;
    track.start_frame = frame_id;
    track.frame_id = frame_id;
    track.vx = 0.0f;
    track.vy = 0.0f;
    track.state = TrackState::Tracked;
    track.is_activated = true;
}

void ByteTracker::reActivateTrack(Track& track, const DetectionRow& det, int frame_id)
{
    const float old_cx = centerX(track);
    const float old_cy = centerY(track);
    const float new_cx = centerX(det);
    const float new_cy = centerY(det);

    track.vx = 0.7f * track.vx + 0.3f * (new_cx - old_cx);
    track.vy = 0.7f * track.vy + 0.3f * (new_cy - old_cy);

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
    const float old_cx = centerX(track);
    const float old_cy = centerY(track);
    const float new_cx = centerX(det);
    const float new_cy = centerY(det);

    track.vx = 0.8f * track.vx + 0.2f * (new_cx - old_cx);
    track.vy = 0.8f * track.vy + 0.2f * (new_cy - old_cy);

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
    float distance_factor,
    std::vector<int>& unmatched_tracks,
    std::vector<int>& unmatched_detections) const
{
    struct Candidate
    {
        int t_idx;
        int d_idx;
        float score;
    };

    std::vector<Candidate> candidates;

    for (size_t t = 0; t < tracks.size(); ++t) {
        const float max_distance = distance_factor * trackSize(tracks[t]);
        for (size_t d = 0; d < detections.size(); ++d) {
            const float iou = computeIoU(detections[d], tracks[t]);
            const float distance = centerDistance(detections[d], tracks[t]);
            if (iou < iou_threshold && distance > max_distance) {
                continue;
            }

            const float score = iou - 0.05f * (distance / max_distance);
            candidates.push_back({ static_cast<int>(t), static_cast<int>(d), score });
        }
    }

    std::sort(
        candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) {
            return a.score > b.score;
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

    std::vector<Track> predicted_tracked;
    predicted_tracked.reserve(tracked_tracks_.size());
    for (const auto& track : tracked_tracks_) {
        predicted_tracked.push_back(predictTrack(track));
    }

    std::vector<int> unmatched_tracked;
    std::vector<int> unmatched_high;
    const float active_iou_thresh = std::max(0.1f, match_thresh_ * 0.5f);
    auto matches_high = greedyMatch(
        predicted_tracked,
        high_dets,
        active_iou_thresh,
        1.8f,
        unmatched_tracked,
        unmatched_high);

    std::vector<Track> new_tracked;
    std::vector<Track> new_lost;
    std::vector<Track> new_removed;
    std::vector<bool> matched_tracked_old(tracked_tracks_.size(), false);
    std::vector<bool> matched_lost_old(lost_tracks_.size(), false);

    for (const auto& match : matches_high) {
        const int track_idx = match.first;
        const int det_idx = match.second;

        Track track = tracked_tracks_[track_idx];
        updateTrack(track, high_dets[det_idx], frame_id_);
        matched_tracked_old[track_idx] = true;
        new_tracked.push_back(track);
    }

    std::vector<Track> remain_tracked;
    std::vector<int> remain_to_original;
    for (size_t i = 0; i < tracked_tracks_.size(); ++i) {
        if (!matched_tracked_old[i]) {
            remain_tracked.push_back(predictTrack(tracked_tracks_[i]));
            remain_to_original.push_back(static_cast<int>(i));
        }
    }

    std::vector<int> unmatched_remain_tracks;
    std::vector<int> unmatched_low;
    auto matches_low = greedyMatch(
        remain_tracked,
        low_dets,
        std::max(0.05f, active_iou_thresh * 0.5f),
        2.2f,
        unmatched_remain_tracks,
        unmatched_low);

    std::vector<bool> low_matched_remain(remain_tracked.size(), false);
    for (const auto& match : matches_low) {
        const int rem_idx = match.first;
        const int det_idx = match.second;

        Track track = tracked_tracks_[remain_to_original[rem_idx]];
        updateTrack(track, low_dets[det_idx], frame_id_);
        matched_tracked_old[remain_to_original[rem_idx]] = true;
        low_matched_remain[rem_idx] = true;
        new_tracked.push_back(track);
    }

    std::vector<DetectionRow> unmatched_high_dets;
    for (int det_idx : unmatched_high) {
        unmatched_high_dets.push_back(high_dets[det_idx]);
    }

    std::vector<Track> predicted_lost;
    predicted_lost.reserve(lost_tracks_.size());
    for (const auto& track : lost_tracks_) {
        predicted_lost.push_back(predictTrack(track));
    }

    std::vector<int> unmatched_lost;
    std::vector<int> still_unmatched_high;
    auto matches_lost = greedyMatch(
        predicted_lost,
        unmatched_high_dets,
        std::max(0.05f, active_iou_thresh * 0.5f),
        2.5f,
        unmatched_lost,
        still_unmatched_high);

    for (const auto& match : matches_lost) {
        const int lost_idx = match.first;
        const int det_idx = match.second;

        Track track = lost_tracks_[lost_idx];
        reActivateTrack(track, unmatched_high_dets[det_idx], frame_id_);
        matched_lost_old[lost_idx] = true;
        new_tracked.push_back(track);
    }

    for (size_t i = 0; i < tracked_tracks_.size(); ++i) {
        if (!matched_tracked_old[i]) {
            Track track = tracked_tracks_[i];
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

    for (int idx : still_unmatched_high) {
        const auto& det = unmatched_high_dets[idx];
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
            if (lost.id == tracked.id || computeIoU(lost, tracked) > 0.85f) {
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
