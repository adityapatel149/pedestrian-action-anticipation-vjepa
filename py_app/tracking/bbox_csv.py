from typing import List, Dict
import csv

from py_app.core.datatypes import Detection

class FramewiseBBoxCSV:
    def __init__(
        self,
        csv_path: str,
        video_id: str,
        frame_width: int,
        frame_height: int,
        max_boxes: int = 10,
    ):
        self.max_boxes = max_boxes
        self.video_id = str(video_id)
        self.by_frame: Dict[int, List[Detection]] = {}

        self.pid_to_track_id: Dict[str, int] = {}
        self.next_track_id = 0

        frame_width_f = float(frame_width)
        frame_height_f = float(frame_height)

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            required = {"video_id", "frame", "participant_id", "x1", "y1", "x2", "y2"}
            missing = required - fields
            if missing:
                raise ValueError(f"CSV missing columns: {sorted(missing)}")

            for row in reader:
                if str(row["video_id"]).strip() != self.video_id:
                    continue

                frame_idx = int(row["frame"])
                pid = str(row["participant_id"]).strip()

                if pid not in self.pid_to_track_id:
                    self.pid_to_track_id[pid] = self.next_track_id
                    self.next_track_id += 1
                track_id = self.pid_to_track_id[pid]

                x1 = float(row["x1"]) / frame_width_f
                y1 = float(row["y1"]) / frame_height_f
                x2 = float(row["x2"]) / frame_width_f
                y2 = float(row["y2"]) / frame_height_f

                det = Detection(
                    track_id=track_id,
                    bbox_xyxy_norm=(x1, y1, x2, y2),
                    score=1.0,
                )
                self.by_frame.setdefault(frame_idx, []).append(det)

        for k, v in self.by_frame.items():
            self.by_frame[k] = v[: self.max_boxes]

        print(
            f"[FramewiseBBoxCSV] video_id={self.video_id}, "
            f"frames_with_boxes={len(self.by_frame)}, "
            f"total_boxes={sum(len(v) for v in self.by_frame.values())}, "
            f"unique_ids={len(self.pid_to_track_id)}"
        )

    def get(self, frame_idx: int) -> List[Detection]:
        return self.by_frame.get(frame_idx, [])