import yaml
import csv
import os

def resolve_video_path_from_csv(csv_path: str, config_path: str) -> str:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    base_path = cfg["experiment"]["data"].get("base_path")
    if not base_path:
        raise ValueError("Config experiment.data.base_path is required to resolve --video from CSV")

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        first = next(reader, None)
        if first is None:
            raise ValueError("CSV is empty")
        if "video_id" not in (reader.fieldnames or []):
            raise ValueError("CSV must contain a video_id column")

        video_id = str(first["video_id"]).strip()
        if not video_id:
            raise ValueError("CSV video_id is empty")

    return os.path.join(base_path, f"{video_id}.mp4")
