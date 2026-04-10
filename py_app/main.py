import torch

from py_app.cli import parse_args
from py_app.core.utils import resolve_video_path_from_csv
from py_app.runners.factory import build_runner
from py_app.core.pipeline import AsyncVideoProcessor


def main():
    args = parse_args()

    if (args.video is None or str(args.video).strip() == "") and args.bbox_csv is not None:
        args.video = resolve_video_path_from_csv(args.bbox_csv, args.config)
        print(f"Resolved video path from CSV/config: {args.video}")

    if args.video is None or str(args.video).strip() == "":
        raise ValueError("A valid --video path is required")

    runner = build_runner(
        config_path=args.config,
        encoder_model=args.encoder_model,
        classifier_model=args.classifier_model,
        sweep_idx=args.sweep_idx,
        device=args.device,
        use_fp16=not args.no_fp16,
    )

    processor = AsyncVideoProcessor(
        runner=runner,
        video_path=args.video,
        output_path=args.output,
        bbox_csv=args.bbox_csv,
        detector_name=args.detector,
        detector_conf=args.detector_conf,
        max_boxes=args.max_boxes,
        display=args.display,
        stride_overlap=args.stride_overlap,
        render_scale=args.render_scale,
        detector_imgsz=args.detector_imgsz,
        anticipation_time=args.anticipation_time,
        tracker_cfg=args.tracker,
        save_bev_video=args.save_bev_video,
        bev_size=args.bev_size,
        use_depth=args.use_depth,
        depth_model=args.depth_model,
        depth_every_n=args.depth_every_n,
        # depth_calib_interval_sec=args.depth_calib_interval_sec,
        # depth_scale_alpha=args.depth_scale_alpha,
        # depth_min_calib_points=args.depth_min_calib_points,
        depth_sample_step=args.depth_sample_step,
        depth_max_points=args.depth_max_points,
        depth_smooth_alpha=args.depth_smooth_alpha,
    )
    processor.run()


if __name__ == "__main__":
    main()