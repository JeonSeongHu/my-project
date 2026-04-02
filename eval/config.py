"""Configuration for NVS Object Completeness Evaluation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EvalConfig:
    # Paths
    data_root: str = "./data"
    output_dir: str = "./results"

    # Dataset
    # Expected structure:
    #   data_root/
    #     scene_001/
    #       input/          <- I_obs (partial view of object)
    #       gt/             <- ground truth novel view (full object visible)
    #       poses/          <- camera poses (input_pose.json, target_pose.json)
    #       masks/          <- object segmentation masks for GT (optional, auto-generated if missing)

    # Models to evaluate
    models: list[str] = field(default_factory=lambda: ["gld", "seva"])

    # Metrics
    compute_psnr: bool = True
    compute_ssim: bool = True
    compute_lpips: bool = True
    compute_completeness: bool = True

    # Segmentation (for completeness metric)
    seg_model: str = "sam"  # "sam" or "mask_provided"
    sam_checkpoint: Optional[str] = None
    sam_model_type: str = "vit_h"

    # Object completeness thresholds
    visibility_threshold: float = 0.5  # IoU threshold to count as "complete"

    # Image settings
    image_size: int = 512
    device: str = "cuda"

    # GLD-specific
    gld_checkpoint: Optional[str] = None
    gld_config: Optional[str] = None

    # Seva-specific
    seva_checkpoint: Optional[str] = None
    seva_config: Optional[str] = None
