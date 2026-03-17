"""
Evaluation / training loop adapted for PIE with four heads:
(cross, action, intersection, signalized).

Assumptions:
- Dataloader yields batches as: (video, bboxes, cross, action, intersection, signalized anticipation_time)
- Classifier forward returns a dict with keys: {"cross": logits, "action": logits, "intersection": logits, "signalized": logits}
- filter_annotations() returns label_encoders for each of cross/action/intersection/signalized and
  trains/vals as (paths, {video_id: df}) tuples.
"""

import os
import logging
import pprint
import random
import time
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import csv 


from evals.action_anticipation_frozen.dataloader import filter_annotations, init_data
from evals.action_anticipation_frozen.losses import sigmoid_focal_loss, softmax_focal_loss
from evals.action_anticipation_frozen.metrics import ClassMeanRecall
from evals.action_anticipation_frozen.models import init_classifier, init_module
from evals.action_anticipation_frozen.utils import init_opt
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.cuda.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)

# FOR PIE
LABEL_KEYS = ("cross", "action", "intersection", "signalized")
HEADS = ("cross", "action", "intersection", "signalized")

class MultiHeadSoftmaxFocal:
    def __init__(self, alpha_by_head, gamma=2.0, reduction="mean"):
        self.alpha_by_head = alpha_by_head
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, head: str, logits: torch.Tensor, targets: torch.Tensor):
        alpha = self.alpha_by_head.get(head, None)
        return softmax_focal_loss(
            logits, targets,
            gamma=self.gamma,
            alpha=alpha,
            reduction=self.reduction,
        )


def main(args_eval, resume_preempt: bool = False):
    test_only = args_eval.get("test_only", False)
    if test_only:
        logger.info("TEST ONLY: running inference and writing CSVs")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # --------------------------- experiment args --------------------------- #
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    test_only = args_eval.get("test_only", False)
    eval_tag = args_eval.get("tag", None)

    # -- PRETRAIN
    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = args_eval.get("experiment")

    # -- CLASSIFIER
    args_classifier = args_exp.get("classifier")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads")

    # -- DATA
    args_data = args_exp.get("data")
    dataset = args_data.get("dataset")  
    base_path = args_data.get("base_path")  
    num_workers = args_data.get("num_workers", 12)
    pin_mem = args_data.get("pin_memory", True)

    frames_per_clip = args_data.get("frames_per_clip")
    frames_per_second = args_data.get("frames_per_second")
    resolution = args_data.get("resolution", 224)

    # anticipation + augs
    train_anticipation_time_sec = args_data.get("train_anticipation_time_sec")
    train_anticipation_point = args_data.get("train_anticipation_point")
    val_anticipation_point = args_data.get("val_anticipation_point", [0.0, 0.0])
    val_anticipation_time_sec = args_data.get("anticipation_time_sec")

    auto_augment = args_data.get("auto_augment")
    motion_shift = args_data.get("motion_shift")
    reprob = args_data.get("reprob")
    random_resize_scale = args_data.get("random_resize_scale")

    train_annotations_path = args_data.get("dataset_train")
    val_annotations_path = args_data.get("dataset_val")
    train_data_path = base_path
    val_data_path = base_path

    framewise_bboxes_csv = args_data.get("framewise_bboxes_csv")

    # -- OPTIMIZATION
    args_opt = args_exp.get("optimization")
    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16")
    use_focal_loss = args_opt.get("use_focal_loss", False)
    use_softmax_focal_loss = args_opt.get("use_softmax_focal_loss", False)
    if args_opt.get("use_focal_loss", False):
        # ignore head name
        criterion = lambda head, logits, targets: sigmoid_focal_loss(logits, targets)
    elif args_opt.get("use_softmax_focal_loss", False):
        alpha_by_head = {
            "cross": torch.tensor([0.25, 0.75], device=device),
            "action": torch.tensor([0.5, 0.5], device=device),
            "intersection": torch.tensor([0.18, 0.25, 0.28, 0.23, 0.16], device=device), # beta=0.99 technique
            "signalized": torch.tensor([0.20, 0.27, 0.29, 0.24], device=device), # # beta=0.99 technique
        }
        criterion = MultiHeadSoftmaxFocal(alpha_by_head, gamma=2.0, reduction="mean")
    else:
        ce = torch.nn.CrossEntropyLoss()
        criterion = lambda head, logits, targets: ce(logits, targets)

    opt_kwargs = [
        dict(
            ref_wd=kwargs.get("weight_decay"),
            final_wd=kwargs.get("final_weight_decay"),
            start_lr=kwargs.get("start_lr"),
            ref_lr=kwargs.get("lr"),
            final_lr=kwargs.get("final_lr"),
            warmup=kwargs.get("warmup"),
        )
        for kwargs in args_opt.get("multihead_kwargs")
    ]

    # ------------------------------ setup --------------------------------- #
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    folder = os.path.join(pretrain_folder, "action_anticipation_frozen/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")

    # CSV logger for three-head setup
    if rank == 0:
        csv_logger = CSVLogger(
            log_file,
            ("%d", "epoch"),
            ("%.5f", "train-acc"),
            ("%.5f", "train-acc-cross"),
            ("%.5f", "train-acc-action"),
            ("%.5f", "train-acc-intersection"),
            ("%.5f", "train-acc-signalized"),
            ("%.5f", "train-recall"),
            ("%.5f", "train-recall-cross"),
            ("%.5f", "train-recall-action"),
            ("%.5f", "train-recall-intersection"),
            ("%.5f", "train-recall-signalized"),
            ("%.5f", "val-acc"),
            ("%.5f", "val-acc-cross"),
            ("%.5f", "val-acc-action"),
            ("%.5f", "val-acc-intersection"),
            ("%.5f", "val-acc-signalized"),
            ("%.5f", "val-recall"),
            ("%.5f", "val-recall-cross"),
            ("%.5f", "val-recall-action"),
            ("%.5f", "val-recall-intersection"),
            ("%.5f", "val-recall-signalized"),
        )

    # -------------------------- annotations ------------------------------- #
    _annotations = filter_annotations(
        dataset=dataset,
        base_path=base_path,
        train_annotations_path=train_annotations_path,
        val_annotations_path=val_annotations_path,
        label_keys=LABEL_KEYS,
    )

    label_encoders: Dict[str, Dict[int, int]] = _annotations["label_encoders"]
    val_label_sets = _annotations["val_label_sets"]
    train_annotations = _annotations["train"]
    val_annotations = _annotations["val"]
    
    print("Train paths:", len(_annotations["train"][0]))
    print("Val paths:", len(_annotations["val"][0]))

    # class cardinalities inferred from encoders
    cross_classes = label_encoders["cross"]
    action_classes = label_encoders["action"]
    intersection_classes = label_encoders["intersection"]
    signalized_classes = label_encoders["signalized"]

    # ------------------------------ model --------------------------------- #
    model = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        frames_per_second=frames_per_second,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )

    classifiers = init_classifier(
        embed_dim=model.embed_dim,
        num_heads=num_heads,
        cross_classes=cross_classes,
        action_classes=action_classes,
        intersection_classes=intersection_classes,
        signalized_classes=signalized_classes,
        num_blocks=num_probe_blocks,
        device=device,
        num_classifiers=len(opt_kwargs),
    )

    # ------------------------------ data ---------------------------------- #
    train_set, train_loader, train_data_info = init_data(
        dataset=dataset,
        training=True,
        base_path=train_data_path,
        annotations_path=train_annotations,
        framewise_bboxes_csv = framewise_bboxes_csv,
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        fps=frames_per_second,
        anticipation_time_sec=train_anticipation_time_sec,
        anticipation_point=train_anticipation_point,
        random_resize_scale=random_resize_scale,
        reprob=reprob,
        auto_augment=auto_augment,
        motion_shift=motion_shift,
        crop_size=resolution,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers,
        pin_mem=pin_mem,
        persistent_workers=False,
    )
    ipe = train_loader.num_batches
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")

    _, val_loader, _ = init_data(
        dataset=dataset,
        training=False,
        base_path=val_data_path,
        annotations_path=val_annotations,
        framewise_bboxes_csv = framewise_bboxes_csv,
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        fps=frames_per_second,
        anticipation_time_sec=val_anticipation_time_sec,
        anticipation_point=val_anticipation_point,
        crop_size=resolution,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers,
        pin_mem=pin_mem,
        persistent_workers=False,
    )
    val_ipe = val_loader.num_batches
    logger.info(f"Val dataloader created... iterations per epoch: {val_ipe}")



    # -------------------------- optimizer --------------------------------- #
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifiers=classifiers,
        opt_kwargs=opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )
    classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]

    # -------------------------- checkpointing ------------------------------ #
    start_epoch = 0
    if resume_checkpoint and os.path.exists(latest_path):
        classifiers, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            classifiers=classifiers,
            opt=optimizer,
            scaler=scaler,
        )
        for _ in range(start_epoch * ipe):
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

    if test_only and os.path.exists(latest_path):
        classifiers, optimizer, scaler, start_epoch = load_checkpoint(device, latest_path, classifiers, optimizer, scaler)
    elif test_only:
        raise RuntimeError(f"No checkpoint found at {latest_path} — cannot run test_only without weights.")

    if test_only:
        out_dir = os.path.join(folder, "inference_csv")
        test_inference(
            device=device,
            model=model,
            classifiers=classifiers,
            data_loader=val_loader,   # or a test_loader if you have one
            use_bfloat16=use_bfloat16,
            encoders=label_encoders,
            out_dir=out_dir,
            split_name="val",
            rank=rank,
        )
        return

    # ------------------------------ train/val ------------------------------ #
    for epoch in range(start_epoch, num_epochs):
        logging.info(f"Epoch {epoch}")
        train_data_info.set_epoch(epoch)

        # ---- train ----
        train_metrics = train_one_epoch(
            ipe=ipe,
            device=device,
            model=model,
            classifiers=classifiers,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=train_loader,
            use_bfloat16=use_bfloat16,
            encoders=label_encoders,
            criterion=criterion,
        )

        # ---- val ----
        val_metrics = validate(
            ipe=val_ipe,
            device=device,
            model=model,
            classifiers=classifiers,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            encoders=label_encoders,
            criterion=criterion,
        )

        logger.info(
            "[%5d]"
            "train acc: %.1f%% (cross %.1f action %.1f intersection %.1f signalized %.1f) "
            "train recall: %.1f%% (cross %.1f action %.1f intersection %.1f signalized %.1f) "
            "val acc: %.1f%% (cross %.1f action %.1f intersection %.1f signalized %.1f) "
            "val recall: %.1f%% (cross %.1f action %.1f intersection %.1f signalized %.1f)"
            % (
                epoch + 1,
                train_metrics["overall"]["accuracy"],
                train_metrics["cross"]["accuracy"],
                train_metrics["action"]["accuracy"],
                train_metrics["intersection"]["accuracy"],
                train_metrics["signalized"]["accuracy"],
                train_metrics["overall"]["recall"],
                train_metrics["cross"]["recall"],
                train_metrics["action"]["recall"],
                train_metrics["intersection"]["recall"],
                train_metrics["signalized"]["recall"],
                val_metrics["overall"]["accuracy"],
                val_metrics["cross"]["accuracy"],
                val_metrics["action"]["accuracy"],
                val_metrics["intersection"]["accuracy"],
                val_metrics["signalized"]["accuracy"],
                val_metrics["overall"]["recall"],
                val_metrics["cross"]["recall"],
                val_metrics["action"]["recall"],
                val_metrics["intersection"]["recall"],
                val_metrics["signalized"]["recall"],
            )
        )

        if rank == 0:
            csv_logger.log(
                epoch + 1,
                train_metrics["overall"]["accuracy"],
                train_metrics["cross"]["accuracy"],
                train_metrics["action"]["accuracy"],
                train_metrics["intersection"]["accuracy"],
                train_metrics["signalized"]["accuracy"],
                train_metrics["overall"]["recall"],
                train_metrics["cross"]["recall"],
                train_metrics["action"]["recall"],
                train_metrics["intersection"]["recall"],
                train_metrics["signalized"]["recall"],
                val_metrics["overall"]["accuracy"],
                val_metrics["cross"]["accuracy"],
                val_metrics["action"]["accuracy"],
                val_metrics["intersection"]["accuracy"],
                val_metrics["signalized"]["accuracy"],
                val_metrics["overall"]["recall"],
                val_metrics["cross"]["recall"],
                val_metrics["action"]["recall"],
                val_metrics["intersection"]["recall"],
                val_metrics["signalized"]["recall"],
            )

        print(f"Saving checkpoints to: {latest_path}")
        save_checkpoint(latest_path, epoch + 1, classifiers, optimizer, scaler, batch_size, world_size)


# --------------------------------- loops ---------------------------------- #

def _build_metric_loggers(num_classes: Dict[str, int], device) -> Dict[str, ClassMeanRecall]:
    return {k: ClassMeanRecall(num_classes=len(v), device=device, k=1) for k, v in num_classes.items()}

def _remap_labels(encoders: Dict[str, Dict[int, int]], labels: Dict[str, torch.Tensor], device):
    remapped = {}
    for k in LABEL_KEYS:
        vals = labels[k]
        # If single int, convert to list
        if isinstance(vals, int):
            vals = [vals]
        elif isinstance(vals, torch.Tensor) and vals.dim() == 0:
            vals = [vals.item()]

        arr = [encoders[k][int(x)] for x in vals]
        t = torch.tensor(arr, device=device, dtype=torch.long)

        remapped[k] = t
    return remapped



def train_one_epoch(
    ipe,
    device,
    model,
    classifiers,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    encoders,
    criterion,
):
    _data_loader = iter(data_loader)
    for c in classifiers:
        c.train(True)

    metric_loggers = _build_metric_loggers(encoders, device)
    data_elapsed_time_meter = AverageMeter()

    for itr in range(ipe):
        itr_start_time = time.time()
        try:
            udata = next(_data_loader)
        except Exception:
            _data_loader = iter(data_loader)
            udata = next(_data_loader)

        [s.step() for s in scheduler]
        [wds.step() for wds in wd_scheduler]

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            # print("Type of udata:", type(udata))
            # print("udata keys (if dict):", getattr(udata, "keys", lambda: None)())

            clips = udata[0].to(device)
            bboxes = udata[1].to(device)
            labels_raw = {"cross": udata[2], "action": udata[3], "intersection": udata[4], "signalized": udata[5]}
            anticipation_times = udata[-1].to(device)
            
            labels = _remap_labels(encoders, labels_raw, device)

            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            with torch.no_grad():
                features = model(clips, anticipation_times)
            outputs = [c(features, bboxes = bboxes) for c in classifiers]

        # compute multi-head loss
        losses = []
        for o in outputs:
            per_head = {h: criterion(h, o[h], labels[h]) for h in HEADS}
            total = per_head["cross"] + 0.5 * (per_head["action"] + per_head["intersection"] + per_head["signalized"])
            losses.append(total)

        if scaler is not None and use_bfloat16:
            [s.scale(L).backward() for s, L in zip(scaler, losses)]
            [s.step(o) for s, o in zip(scaler, optimizer)]
            [s.update() for s in scaler]
        else:
            [L.backward() for L in losses]
            [o.step() for o in optimizer]
        [o.zero_grad() for o in optimizer]

        # metrics
        with torch.no_grad():
            cross_metrics = [metric_loggers["cross"](o["cross"], labels["cross"]) for o in outputs]
            action_metrics = [metric_loggers["action"](o["action"], labels["action"]) for o in outputs]
            intersection_metrics = [metric_loggers["intersection"](o["intersection"], labels["intersection"]) for o in outputs]
            signalized_metrics = [metric_loggers["signalized"](o["signalized"], labels["signalized"]) for o in outputs]

        if itr % 10 == 0 or itr == ipe - 1:
            logger.info(
                "[%5d] acc: %.1f (cross %.1f act %.1f int %.1f sig %.1f) "
                "recall: %.1f (cross %.1f act %.1f int %.1f sig %.1f) [mem: %.2e] [data: %.1f ms]"
                % (
                    itr,
                    max([c["accuracy"] for c in cross_metrics]),
                    max([c["accuracy"] for c in cross_metrics]),
                    max([a["accuracy"] for a in action_metrics]),
                    max([i["accuracy"] for i in intersection_metrics]),
                    max([s["accuracy"] for s in signalized_metrics]),
                    max([c["recall"] for c in cross_metrics]),
                    max([c["recall"] for c in cross_metrics]),
                    max([a["recall"] for a in action_metrics]),
                    max([i["recall"] for i in intersection_metrics]),
                    max([s["recall"] for s in signalized_metrics]),
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                    data_elapsed_time_meter.avg,
                )
            )

    del _data_loader

    def _summarize(mets):
        return dict(accuracy=max([m["accuracy"] for m in mets]), recall=max([m["recall"] for m in mets]))

    ret = {
        "overall": _summarize(cross_metrics),
        "cross": _summarize(cross_metrics),
        "action": _summarize(action_metrics),
        "intersection": _summarize(intersection_metrics),
        "signalized": _summarize(signalized_metrics),
    }
    return ret


@torch.no_grad()

def validate(
    ipe,
    device,
    model,
    classifiers,
    data_loader,
    use_bfloat16,
    encoders,
    criterion,
):
    logger.info("Running val...")
    _data_loader = iter(data_loader)
    for c in classifiers:
        c.train(False)

    metric_loggers = _build_metric_loggers(encoders, device)

    for itr in range(ipe):
        try:
            udata = next(_data_loader)
        except Exception:
            _data_loader = iter(data_loader)
            udata = next(_data_loader)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            
            clips = udata[0].to(device)
            bboxes = udata[1].to(device)
            labels_raw = {"cross": udata[2], "action": udata[3], "intersection": udata[4], "signalized": udata[5]}
            anticipation_times = udata[-1].to(device)
            
            labels = _remap_labels(encoders, labels_raw, device)

            features = model(clips, anticipation_times)
            outputs = [c(features, bboxes = bboxes) for c in classifiers]


            loss = 0.0
            for o in outputs:
                per_head = {h: criterion(h, o[h], labels[h]) for h in HEADS}
                loss = loss + (per_head["cross"] + 0.5 * (per_head["action"] + per_head["intersection"] + per_head["signalized"]))

               

            cross_metrics = [metric_loggers["cross"](o["cross"], labels["cross"]) for o in outputs]
            action_metrics = [metric_loggers["action"](o["action"], labels["action"]) for o in outputs]
            intersection_metrics = [metric_loggers["intersection"](o["intersection"], labels["intersection"]) for o in outputs]
            signalized_metrics = [metric_loggers["signalized"](o["signalized"], labels["signalized"]) for o in outputs]

        if itr % 10 == 0 or itr == ipe - 1:
            logger.info(
                "[%5d] acc: %.1f (cross %.1f act %.1f int %.1f sig %.1f) "
                "recall: %.1f (cross %.1f act %.1f int %.1f sig %.1f) "
                "loss: %.3f [mem: %.2e]"
                % (
                    itr,
                    max([c["accuracy"] for c in cross_metrics]),
                    max([c["accuracy"] for c in cross_metrics]),
                    max([a["accuracy"] for a in action_metrics]),
                    max([i["accuracy"] for i in intersection_metrics]),
                    max([s["accuracy"] for s in signalized_metrics]),
                    max([c["recall"] for c in cross_metrics]),
                    max([c["recall"] for c in cross_metrics]),
                    max([a["recall"] for a in action_metrics]),
                    max([i["recall"] for i in intersection_metrics]),
                    max([s["recall"] for s in signalized_metrics]),
                    loss,
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                )
            )

    del _data_loader

    def _summarize(mets):
        return dict(accuracy=max([m["accuracy"] for m in mets]), recall=max([m["recall"] for m in mets]))

    ret = {
        "overall": _summarize(cross_metrics),
        "cross": _summarize(cross_metrics),
        "action": _summarize(action_metrics),
        "intersection": _summarize(intersection_metrics),
        "signalized": _summarize(signalized_metrics),
    }
    return ret


@torch.no_grad()
def test_inference(
    device,
    model,
    classifiers,
    data_loader,
    use_bfloat16,
    encoders,
    out_dir,
    split_name="test",
    rank=0,
):
    """
    Runs inference for each classifier sweep and writes separate CSVs.

    Output files (one per sweep, per rank):
      {out_dir}/{split_name}_sweep{j}_r{rank}.csv

    - classifiers: list of classifier modules; each classifier's forward returns a dict with keys in HEADS
    - encoders: label_encoders from filter_annotations()
    - data_loader: DataLoader yielding batches of the form
        (clips, bboxes, cross, action, intersection, signalized, anticipation_time)
    """
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    for c in classifiers:
        c.train(False)  # ensure eval mode for classifiers

    writers = []
    files = []
    num_sweeps = len(classifiers)

    # We don’t know num_classes until first forward; init lazily
    header_written = [False] * num_sweeps
    header_cols = [None] * num_sweeps  # store header columns for each sweep

    try:
        for batch_idx, udata in enumerate(data_loader):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                clips = udata[0].to(device)
                bboxes = udata[1].to(device)
                # assume loader yields: (clips, bboxes, cross, action, intersection, signalized, anticipation_time)
                labels_raw = {
                    "cross": udata[2],
                    "action": udata[3],
                    "intersection": udata[4],
                    "signalized": udata[5],
                }
                anticipation_times = udata[-1].to(device)

                labels = _remap_labels(encoders, labels_raw, device)  # remapped y_true per head

                features = model(clips, anticipation_times)  # trunk frozen, returns features

                # open CSV files lazily on first batch (so we can get shapes later)
                if not files:
                    for j in range(num_sweeps):
                        path = os.path.join(out_dir, f"{split_name}_sweep{j}_r{rank}.csv")
                        f = open(path, "w", newline="")
                        files.append(f)
                        writers.append(csv.writer(f))

                # run each sweep
                for j, clf in enumerate(classifiers):
                    out = clf(features, bboxes=bboxes)  # out is dict with possible keys in HEADS

                    # Determine header when first encountering this sweep
                    if not header_written[j]:
                        header = ["batch_idx", "sample_idx", "anticipation_time"]
                        for h in HEADS:
                            logits_h = out.get(h, None)
                            if logits_h is None:
                                continue
                            C = logits_h.shape[1]
                            header += [f"y_true_{h}", f"pred_{h}"]
                            header += [f"p_{h}_{k}" for k in range(C)]
                            header += [f"logit_{h}_{k}" for k in range(C)]
                        writers[j].writerow(header)
                        header_written[j] = True
                        header_cols[j] = header

                    # Compute probs/preds for present heads
                    probs_by_head = {}
                    preds_by_head = {}
                    logits_by_head = {}
                    for h in HEADS:
                        logits_h = out.get(h, None)
                        if logits_h is None:
                            probs_by_head[h] = None
                            preds_by_head[h] = None
                            logits_by_head[h] = None
                            continue
                        probs = F.softmax(logits_h, dim=-1)
                        pred = probs.argmax(dim=-1)
                        probs_by_head[h] = probs
                        preds_by_head[h] = pred
                        logits_by_head[h] = logits_h

                    B = features.shape[0]
                    for i in range(B):
                        row = [batch_idx, i, float(anticipation_times[i].item())]
                        for h in HEADS:
                            logits_h = logits_by_head.get(h, None)
                            probs_h = probs_by_head.get(h, None)
                            pred_h = preds_by_head.get(h, None)
                            y_true_tensor = labels.get(h, None)

                            if logits_h is None or probs_h is None:
                                # Add placeholders for y_true and pred to keep alignment
                                row += [None, None]
                                continue

                            # Safely extract y_true if possible
                            try:
                                y_true_val = int(y_true_tensor[i].item()) if (y_true_tensor is not None and y_true_tensor.numel() > i) else None
                            except Exception:
                                y_true_val = None

                            row.append(y_true_val)
                            row.append(int(pred_h[i].item()))
                            # probs and logits columns
                            row += [float(probs_h[i, k].item()) for k in range(probs_h.shape[1])]
                            row += [float(logits_h[i, k].item()) for k in range(logits_h.shape[1])]

                        writers[j].writerow(row)

            if batch_idx % 10 == 0:
                logger.info(f"[test] rank {rank} processed batch {batch_idx}")

    finally:
        for f in files:
            try:
                f.close()
            except Exception:
                pass

    logger.info(f"[test] Wrote {num_sweeps} CSV(s) to: {out_dir}")
# ---------------------------- checkpoint I/O ------------------------------ #

def load_checkpoint(device, r_path, classifiers, opt, scaler):
    logger.info(f"read-path: {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    msg = [c.load_state_dict(pd) for c, pd in zip(classifiers, checkpoint["classifiers"])]
    logger.info(f"loaded pretrained classifier with msg: {msg}")
    [o.load_state_dict(c) for o, c in zip(opt, checkpoint["opt"])]
    if scaler is not None and checkpoint.get("scaler") is not None:
        [s.load_state_dict(c) for s, c in zip(scaler, checkpoint["scaler"])]
    epoch = checkpoint.get("epoch", 0)
    return classifiers, opt, scaler, epoch


def save_checkpoint(latest_path, epoch, classifiers, optimizer, scaler, batch_size, world_size):
    save_dict = {
        "classifiers": [c.state_dict() for c in classifiers],
        "opt": [o.state_dict() for o in optimizer],
        "scaler": None if scaler is None else [s.state_dict() for s in scaler],
        "epoch": epoch,
        "batch_size": batch_size,
        "world_size": world_size,
    }
    torch.save(save_dict, latest_path)
    torch.save(save_dict, f"{latest_path}_epoch{epoch}.pt")
