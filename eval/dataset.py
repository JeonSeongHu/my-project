"""Dataset loader for NVS Object Completeness Benchmark.

Expected directory structure:
    data_root/
      scene_001/
        input/
          image.png           <- I_obs: partial view of object A
        gt/
          image.png           <- GT novel view: object A fully visible
        poses/
          input_pose.json     <- camera extrinsics/intrinsics for input view
          target_pose.json    <- camera extrinsics/intrinsics for target view
        masks/
          gt_mask.png         <- (optional) object segmentation mask on GT view
          input_mask.png      <- (optional) object segmentation mask on input view
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class SceneSample:
    scene_id: str
    input_image: torch.Tensor       # [3, H, W] partial view
    gt_image: torch.Tensor          # [3, H, W] full view (ground truth)
    input_pose: dict                 # camera pose for input
    target_pose: dict                # camera pose for target (novel view)
    gt_mask: torch.Tensor | None     # [1, H, W] object mask on GT
    input_mask: torch.Tensor | None  # [1, H, W] object mask on input
    input_visibility: float          # fraction of object visible in input (0~1)


def load_pose(pose_path: Path) -> dict:
    """Load camera pose from JSON.

    Expected format:
    {
        "extrinsic": [[4x4 matrix]],   <- world-to-camera
        "intrinsic": [[3x3 matrix]],   <- camera intrinsic
        "width": int,
        "height": int
    }
    """
    with open(pose_path) as f:
        return json.load(f)


class NVSCompletenessDataset(Dataset):
    """Dataset for evaluating object completeness in NVS."""

    def __init__(self, data_root: str, image_size: int = 512):
        self.data_root = Path(data_root)
        self.image_size = image_size

        # Discover scenes
        self.scenes = sorted([
            d for d in self.data_root.iterdir()
            if d.is_dir() and (d / "input").exists() and (d / "gt").exists()
        ])

        if not self.scenes:
            raise ValueError(f"No valid scenes found in {data_root}. "
                             "Each scene needs input/ and gt/ subdirectories.")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.scenes)

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def _load_mask(self, path: Path) -> torch.Tensor | None:
        if not path.exists():
            return None
        mask = Image.open(path).convert("L")
        mask_tensor = self.mask_transform(mask)
        return (mask_tensor > 0.5).float()

    def _find_image(self, directory: Path) -> Path:
        """Find the first image file in directory."""
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            candidates = list(directory.glob(f"*{ext}"))
            if candidates:
                return candidates[0]
        raise FileNotFoundError(f"No image found in {directory}")

    def _compute_input_visibility(
        self, input_mask: torch.Tensor | None, gt_mask: torch.Tensor | None
    ) -> float:
        """Estimate how much of the object is visible in input vs GT.

        Returns ratio of visible object pixels in input / total object pixels in GT.
        If masks not available, returns -1 (unknown).
        """
        if input_mask is None or gt_mask is None:
            return -1.0
        input_area = input_mask.sum().item()
        gt_area = gt_mask.sum().item()
        if gt_area == 0:
            return 1.0
        return min(input_area / gt_area, 1.0)

    def __getitem__(self, idx: int) -> SceneSample:
        scene_dir = self.scenes[idx]
        scene_id = scene_dir.name

        # Load images
        input_image = self._load_image(self._find_image(scene_dir / "input"))
        gt_image = self._load_image(self._find_image(scene_dir / "gt"))

        # Load poses
        pose_dir = scene_dir / "poses"
        input_pose = load_pose(pose_dir / "input_pose.json") if (pose_dir / "input_pose.json").exists() else {}
        target_pose = load_pose(pose_dir / "target_pose.json") if (pose_dir / "target_pose.json").exists() else {}

        # Load masks (optional)
        mask_dir = scene_dir / "masks"
        gt_mask = self._load_mask(mask_dir / "gt_mask.png") if mask_dir.exists() else None
        input_mask = self._load_mask(mask_dir / "input_mask.png") if mask_dir.exists() else None

        input_visibility = self._compute_input_visibility(input_mask, gt_mask)

        return SceneSample(
            scene_id=scene_id,
            input_image=input_image,
            gt_image=gt_image,
            input_pose=input_pose,
            target_pose=target_pose,
            gt_mask=gt_mask,
            input_mask=input_mask,
            input_visibility=input_visibility,
        )
