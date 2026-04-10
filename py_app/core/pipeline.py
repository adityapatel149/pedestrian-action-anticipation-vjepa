from typing import Optional, Tuple, Any
import cv2
import os
import queue
import threading

from py_app.core.datatypes import EngineOutput
from py_app.core.streaming_anticipation_engine import StreamingAnticipationEngine
from py_app.tracking.tracking_node import TrackingNode
from py_app.visualization.visualization_node import VisualizationNode


class AsyncVideoProcessor:
    def __init__(
        self,
        runner,
        video_path: Optional[str],
        output_path: Optional[str] = None,
        save_bev_video: bool = False,
        bev_size=None,
        bbox_csv: Optional[str] = None,
        detector_name: str = "yolo26n.pt",
        detector_conf: float = 0.25,
        max_boxes: int = 10,
        display: bool = True,
        stride_overlap: float = 0.30,
        render_scale: float = 1.0,
        detector_imgsz: int = 416,
        anticipation_time: float = 1.0,
        tracker_cfg: str = "bytetrack.yaml",
        use_depth: bool = False,
        depth_model: Optional[str] = None,
        depth_every_n: int = 3,
        depth_calib_interval_sec: float = 3.0,
        depth_scale_alpha: float = 0.15,
        depth_min_calib_points: int = 2,
        depth_sample_step: int = 12,
        depth_max_points: int = 12000,
        depth_smooth_alpha: float = 0.2,
    ):
        self.runner = runner
        self.video_path = video_path
        self.display = bool(display)
        self.render_scale = float(render_scale)
        self.output_path = output_path
        self.save_bev_video = bool(save_bev_video)

        if video_path is None or str(video_path).strip() == "":
            raise ValueError("A valid video path is required")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        self.src_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out_width = max(1, int(round(self.width * self.render_scale)))
        self.out_height = max(1, int(round(self.height * self.render_scale)))

        if self.save_bev_video and self.output_path is None:
            raise ValueError("output_path is required when save_bev_video=True")

        self.enable_overlay_render = self.display or (self.output_path is not None)
        self.enable_bev_render = self.display or self.save_bev_video

        self.tracking_node = TrackingNode(
            runner=runner,
            frame_width=self.width,
            frame_height=self.height,
            bbox_csv=bbox_csv,
            video_path=video_path,
            detector_name=detector_name,
            detector_conf=detector_conf,
            max_boxes=max_boxes,
            detector_imgsz=detector_imgsz,
            tracker_cfg=tracker_cfg,
        )

        self.visualization_node = VisualizationNode(
            frame_width=self.width,
            frame_height=self.height,
            src_fps=self.src_fps,
            use_depth=use_depth,
            depth_model=depth_model,
            depth_every_n=depth_every_n,
            depth_calib_interval_sec=depth_calib_interval_sec,
            depth_scale_alpha=depth_scale_alpha,
            depth_min_calib_points=depth_min_calib_points,
            depth_sample_step=depth_sample_step,
            depth_max_points=depth_max_points,
            depth_smooth_alpha=depth_smooth_alpha,
            bev_size=bev_size,
            enable_overlay_render=self.enable_overlay_render,
            enable_bev_render=self.enable_bev_render,
            device=str(runner.device),
        )

        self.engine = StreamingAnticipationEngine(
            runner=runner,
            src_fps=self.src_fps,
            frame_width=self.width,
            frame_height=self.height,
            max_boxes=max_boxes,
            stride_overlap=stride_overlap,
            anticipation_time=anticipation_time,
            camera_config=self.visualization_node.camera_config,
            bev_config=self.visualization_node.bev_config,
        )

        self.overlay_writer = None
        self.bev_writer = None
        self.bev_output_path: Optional[str] = None
        self._setup_writers()

        self.read_queue: "queue.Queue[Optional[Tuple[int, Any]]]" = queue.Queue(maxsize=128)
        self.stop_event = threading.Event()

        self.current_frame_idx: Optional[int] = None
        self.current_frame = None

    def _setup_writers(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        if self.output_path is not None:
            out_dir = os.path.dirname(self.output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            self.overlay_writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.src_fps,
                (self.out_width, self.out_height),
            )
            if not self.overlay_writer.isOpened():
                raise RuntimeError(f"Failed to open overlay writer: {self.output_path}")

        if self.save_bev_video:
            base, ext = os.path.splitext(self.output_path)
            if ext == "":
                ext = ".mp4"
            self.bev_output_path = f"{base}_bev{ext}"

            bev_dir = os.path.dirname(self.bev_output_path)
            if bev_dir:
                os.makedirs(bev_dir, exist_ok=True)

            bev_size_px = self.visualization_node.bev_config.bev_size
            self.bev_writer = cv2.VideoWriter(
                self.bev_output_path,
                fourcc,
                self.src_fps,
                (bev_size_px, bev_size_px),
            )
            if not self.bev_writer.isOpened():
                raise RuntimeError(f"Failed to open BEV writer: {self.bev_output_path}")

    def read_loop(self):
        frame_idx = 0
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                break
            self.read_queue.put((frame_idx, frame))
            frame_idx += 1
        self.read_queue.put(None)

    def handle_result(self, result: EngineOutput) -> bool:
        pred_map = {p.track_id: p for p in result.predictions}

        self.visualization_node.update_depth_and_bev(
            frame_idx=self.current_frame_idx,
            frame_bgr=self.current_frame,
            detections=result.detections,
            predictions=pred_map,
        )

        rendered = self.visualization_node.render_overlay(
            frame_bgr=self.current_frame,
            detections=result.detections,
            predictions=pred_map,
        )
        bev = self.visualization_node.last_bev

        if rendered is not None and (self.out_width, self.out_height) != (self.width, self.height):
            rendered = cv2.resize(
                rendered,
                (self.out_width, self.out_height),
                interpolation=cv2.INTER_LINEAR,
            )

        if rendered is not None and self.overlay_writer is not None:
            self.overlay_writer.write(rendered)

        if bev is not None and self.bev_writer is not None:
            self.bev_writer.write(bev)

        if self.display:
            if rendered is not None:
                cv2.imshow("crossing-demo", rendered)
            if bev is not None:
                cv2.imshow("bev-demo", bev)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                self.stop_event.set()
                return False

        return True

    def run(self):
        t_reader = threading.Thread(target=self.read_loop, daemon=True)
        t_reader.start()

        try:
            while True:
                item = self.read_queue.get()
                if item is None:
                    break

                frame_idx, frame = item
                self.current_frame_idx = frame_idx
                self.current_frame = frame

                detections = self.tracking_node.get_detections(frame_idx, frame)

                result = self.engine.process_frame(
                    frame_bgr=frame,
                    frame_idx=frame_idx,
                    detections=detections,
                )

                if not self.handle_result(result):
                    break
        finally:
            t_reader.join(timeout=1.0)
            self.cap.release()

            if self.overlay_writer is not None:
                self.overlay_writer.release()

            if self.bev_writer is not None:
                self.bev_writer.release()

            if self.display:
                cv2.destroyAllWindows()