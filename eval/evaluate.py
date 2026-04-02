#!/usr/bin/env python3
"""Main evaluation script for NVS Object Completeness Benchmark.

Usage:
    # Evaluate all models (GLD + Seva)
    python eval/evaluate.py --data_root ./data --output_dir ./results

    # Evaluate specific model
    python eval/evaluate.py --data_root ./data --models gld

    # With SAM for object segmentation
    python eval/evaluate.py --data_root ./data --sam_checkpoint ./sam_vit_h.pth

    # CPU-only
    python eval/evaluate.py --data_root ./data --device cpu
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import asdict

import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image

from config import EvalConfig
from dataset import NVSCompletenessDataset, SceneSample
from metrics import MetricSuite, MetricResult
from models import build_model


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="NVS Object Completeness Evaluation")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--models", nargs="+", default=["gld", "seva"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=512)

    # Segmentation
    parser.add_argument("--seg_model", type=str, default="sam",
                        choices=["sam", "mask_provided"])
    parser.add_argument("--sam_checkpoint", type=str, default=None)
    parser.add_argument("--sam_model_type", type=str, default="vit_h")

    # Model checkpoints
    parser.add_argument("--gld_checkpoint", type=str, default=None)
    parser.add_argument("--gld_config", type=str, default=None)
    parser.add_argument("--seva_checkpoint", type=str, default=None)
    parser.add_argument("--seva_config", type=str, default=None)

    args = parser.parse_args()

    config = EvalConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        models=args.models,
        device=args.device,
        image_size=args.image_size,
        seg_model=args.seg_model,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        gld_checkpoint=args.gld_checkpoint,
        gld_config=args.gld_config,
        seva_checkpoint=args.seva_checkpoint,
        seva_config=args.seva_config,
    )
    return config


def save_visualization(
    output_dir: Path,
    scene_id: str,
    model_name: str,
    input_img: torch.Tensor,
    pred_img: torch.Tensor,
    gt_img: torch.Tensor,
    gt_mask: torch.Tensor | None = None,
):
    """Save side-by-side visualization: input | predicted | GT."""
    vis_dir = output_dir / "visualizations" / model_name
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Compose: [input | pred | GT]
    images = [input_img, pred_img, gt_img]

    if gt_mask is not None:
        # Add masked pred overlay
        mask_3c = gt_mask.expand_as(pred_img)
        masked_pred = pred_img * mask_3c + (1 - mask_3c) * 0.3
        images.append(masked_pred)

    grid = torch.cat(images, dim=2)  # concatenate along width
    save_image(grid, vis_dir / f"{scene_id}.png")


def evaluate_model(
    model_name: str,
    config: EvalConfig,
    dataset: NVSCompletenessDataset,
    metric_suite: MetricSuite,
) -> list[MetricResult]:
    """Evaluate a single NVS model on the entire dataset."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name.upper()}")
    print(f"{'='*60}")

    # Build and load model
    model = build_model(model_name)
    ckpt = getattr(config, f"{model_name}_checkpoint", None)
    cfg = getattr(config, f"{model_name}_config", None)
    model.load(checkpoint=ckpt, config=cfg, device=config.device)

    output_dir = Path(config.output_dir)
    results = []

    for i in range(len(dataset)):
        sample: SceneSample = dataset[i]
        print(f"\n  [{i+1}/{len(dataset)}] Scene: {sample.scene_id} "
              f"(input visibility: {sample.input_visibility:.1%})")

        # Run NVS
        pred_image = model.synthesize(
            input_image=sample.input_image.to(config.device),
            input_pose=sample.input_pose,
            target_pose=sample.target_pose,
        )
        pred_image = pred_image.cpu()

        # Compute metrics
        result = metric_suite.evaluate(
            pred_image=pred_image,
            gt_image=sample.gt_image,
            gt_mask=sample.gt_mask,
            scene_id=sample.scene_id,
            model_name=model_name,
            input_visibility=sample.input_visibility,
        )
        results.append(result)

        print(f"    PSNR={result.psnr:.2f}  SSIM={result.ssim:.4f}  LPIPS={result.lpips:.4f}")
        print(f"    ObjPSNR={result.obj_psnr:.2f}  OCS={result.object_completeness_score:.4f}  "
              f"IoU={result.mask_iou:.4f}")

        # Save visualization
        save_visualization(
            output_dir=output_dir,
            scene_id=sample.scene_id,
            model_name=model_name,
            input_img=sample.input_image,
            pred_img=pred_image,
            gt_img=sample.gt_image,
            gt_mask=sample.gt_mask,
        )

    return results


def aggregate_results(all_results: dict[str, list[MetricResult]]) -> dict:
    """Aggregate per-scene results into summary statistics."""
    summary = {}

    for model_name, results in all_results.items():
        if not results:
            continue

        metrics = {
            "psnr": [r.psnr for r in results],
            "ssim": [r.ssim for r in results],
            "lpips": [r.lpips for r in results],
            "obj_psnr": [r.obj_psnr for r in results if r.obj_psnr >= 0],
            "obj_ssim": [r.obj_ssim for r in results if r.obj_ssim >= 0],
            "obj_lpips": [r.obj_lpips for r in results if r.obj_lpips >= 0],
            "ocs": [r.object_completeness_score for r in results],
            "mask_iou": [r.mask_iou for r in results],
        }

        summary[model_name] = {}
        for key, values in metrics.items():
            if values:
                summary[model_name][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

        # Breakdown by input visibility bins
        bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
        visibility_breakdown = {}
        for lo, hi in bins:
            bin_results = [r for r in results if lo <= r.input_visibility < hi]
            if bin_results:
                visibility_breakdown[f"{lo:.0%}-{hi:.0%}"] = {
                    "count": len(bin_results),
                    "ocs_mean": float(np.mean([r.object_completeness_score for r in bin_results])),
                    "psnr_mean": float(np.mean([r.psnr for r in bin_results])),
                    "mask_iou_mean": float(np.mean([r.mask_iou for r in bin_results])),
                }
        summary[model_name]["by_input_visibility"] = visibility_breakdown

    return summary


def print_summary(summary: dict):
    """Pretty-print evaluation summary."""
    print(f"\n{'='*70}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*70}")

    for model_name, metrics in summary.items():
        print(f"\n  Model: {model_name.upper()}")
        print(f"  {'-'*50}")

        # Full-image metrics
        for key in ["psnr", "ssim", "lpips"]:
            if key in metrics:
                m = metrics[key]
                arrow = "↑" if key != "lpips" else "↓"
                print(f"    {key.upper():>10s} {arrow}: {m['mean']:.4f} ± {m['std']:.4f}")

        print()
        # Object-region metrics
        for key in ["obj_psnr", "obj_ssim", "obj_lpips"]:
            if key in metrics:
                m = metrics[key]
                arrow = "↑" if "lpips" not in key else "↓"
                print(f"    {key.upper():>10s} {arrow}: {m['mean']:.4f} ± {m['std']:.4f}")

        print()
        # Completeness
        for key in ["ocs", "mask_iou"]:
            if key in metrics:
                m = metrics[key]
                print(f"    {key.upper():>10s} ↑: {m['mean']:.4f} ± {m['std']:.4f}")

        # Visibility breakdown
        if "by_input_visibility" in metrics:
            print(f"\n    By Input Visibility:")
            for bin_name, stats in metrics["by_input_visibility"].items():
                print(f"      {bin_name:>10s}: OCS={stats['ocs_mean']:.4f}  "
                      f"PSNR={stats['psnr_mean']:.2f}  IoU={stats['mask_iou_mean']:.4f}  "
                      f"(n={stats['count']})")


def main():
    config = parse_args()

    # Setup output
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check device
    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        config.device = "cpu"

    # Load dataset
    print(f"Loading dataset from: {config.data_root}")
    dataset = NVSCompletenessDataset(
        data_root=config.data_root,
        image_size=config.image_size,
    )
    print(f"Found {len(dataset)} scenes.")

    # Initialize metrics
    metric_suite = MetricSuite(
        device=config.device,
        seg_method=config.seg_model,
        sam_checkpoint=config.sam_checkpoint,
        sam_model_type=config.sam_model_type,
    )

    # Evaluate each model
    all_results: dict[str, list[MetricResult]] = {}

    for model_name in config.models:
        results = evaluate_model(model_name, config, dataset, metric_suite)
        all_results[model_name] = results

        # Save per-scene results
        model_results_file = output_dir / f"{model_name}_results.json"
        with open(model_results_file, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"  Saved per-scene results to {model_results_file}")

    # Aggregate and print summary
    summary = aggregate_results(all_results)
    print_summary(summary)

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_file}")

    # Save config
    config_file = output_dir / "eval_config.json"
    with open(config_file, "w") as f:
        json.dump(asdict(config), f, indent=2)


if __name__ == "__main__":
    main()
