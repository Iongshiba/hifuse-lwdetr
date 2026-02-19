# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train, eval, and inference functions used in main.py
"""
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from util.misc import nested_tensor_from_tensor_list


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    ema_m: torch.nn.Module = None,
    schedules: dict = {},
    num_training_steps_per_epoch=None,
    vit_encoder_num_layers=None,
    args=None,
):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    start_steps = epoch * num_training_steps_per_epoch

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        it = start_steps + data_iter_step
        if "dp" in schedules:
            if args.distributed:
                model.module.update_drop_path(
                    schedules["dp"][it], vit_encoder_num_layers
                )
            else:
                model.update_drop_path(schedules["dp"][it], vit_encoder_num_layers)
        if "do" in schedules:
            if args.distributed:
                model.module.update_dropout(schedules["do"][it])
            else:
                model.update_dropout(schedules["do"][it])

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples, targets)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        if ema_m is not None:
            if epoch >= 0:
                ema_m.update(model)
        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args=None):
    model.eval()
    if args.fp16_eval:
        model.half()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if args.fp16_eval:
            samples.tensors = samples.tensors.half()

        outputs = model(samples)

        if args.fp16_eval:
            for key in outputs.keys():
                if key == "enc_outputs":
                    for sub_key in outputs[key].keys():
                        outputs[key][sub_key] = outputs[key][sub_key].float()
                elif key == "aux_outputs":
                    for idx in range(len(outputs[key])):
                        for sub_key in outputs[key][idx].keys():
                            outputs[key][idx][sub_key] = outputs[key][idx][
                                sub_key
                            ].float()
                else:
                    outputs[key] = outputs[key].float()

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in postprocessors.keys():
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in postprocessors.keys():
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    return stats, coco_evaluator


def get_inference_transform(image_size=224):
    """Get inference transforms matching training transforms"""
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            normalize,
        ]
    )


@torch.no_grad()
def inference(
    model,
    postprocessors,
    input_path,
    output_dir,
    device,
    score_threshold=0.5,
    visualize=False,
    save_json=False,
    class_names=None,
    image_size=224,
):
    model.eval()

    # Setup transforms
    transform = get_inference_transform(image_size)

    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get input images
    input_path = Path(input_path)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = (
            list(input_path.glob("*.jpg"))
            + list(input_path.glob("*.jpeg"))
            + list(input_path.glob("*.png"))
            + list(input_path.glob("*.bmp"))
        )
    else:
        raise ValueError(f"Input path {input_path} does not exist")

    print(f"Found {len(image_paths)} images to process")

    all_results = {}
    total_time = 0

    for image_path in image_paths:
        print(f"Processing {image_path.name}...")

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        orig_size = image.size  # (width, height)
        image_tensor = transform(image)

        # Run inference
        start_time = time.time()
        samples = nested_tensor_from_tensor_list([image_tensor.to(device)])
        outputs = model(samples)

        # Use postprocessor
        orig_target_sizes = torch.tensor([[orig_size[1], orig_size[0]]], device=device)
        results = postprocessors["bbox"](outputs, orig_target_sizes)[0]

        inference_time = time.time() - start_time
        total_time += inference_time

        # Filter by score threshold
        keep = results["scores"] >= score_threshold
        scores = results["scores"][keep].cpu().numpy()
        labels = results["labels"][keep].cpu().numpy()
        boxes = results["boxes"][keep].cpu().numpy()

        print(
            f"  Found {len(scores)} detections (inference time: {inference_time:.3f}s)"
        )

        # Save results
        result = {
            "image": str(image_path),
            "inference_time": inference_time,
            "detections": [],
        }

        for score, label, box in zip(scores, labels, boxes):
            result["detections"].append(
                {
                    "score": float(score),
                    "label": int(label),
                    "box": [float(x) for x in box],
                }
            )

        all_results[image_path.name] = result

        # Visualize if requested
        if visualize:
            vis_image = image.copy()
            draw = ImageDraw.Draw(vis_image)

            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
                )
            except:
                font = ImageFont.load_default()

            colors = [
                "red",
                "green",
                "blue",
                "yellow",
                "purple",
                "orange",
                "cyan",
                "magenta",
            ]

            for score, label, box in zip(scores, labels, boxes):
                color = colors[int(label) % len(colors)]
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

                if class_names and int(label) < len(class_names):
                    label_text = f"{class_names[int(label)]}: {score:.2f}"
                else:
                    label_text = f"Class {int(label)}: {score:.2f}"

                draw.text((x1, y1 - 15), label_text, fill=color, font=font)

            vis_image.save(output_dir / f"{image_path.stem}_pred.jpg")

    # Save JSON results
    if save_json:
        json_path = output_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {json_path}")

    # Print summary
    print(f"\nInference complete!")
    print(f"  Total images: {len(image_paths)}")
    print(f"  Total time: {total_time:.3f}s")
    if len(image_paths) > 0:
        print(f"  Average time per image: {total_time / len(image_paths):.3f}s")
    print(f"  Output directory: {output_dir}")

    return all_results
