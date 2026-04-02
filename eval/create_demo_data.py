#!/usr/bin/env python3
"""Generate synthetic demo data for testing the evaluation pipeline.

Creates fake scenes with:
- Input image: object partially visible (cropped)
- GT image: object fully visible
- Object masks
- Camera poses

Usage:
    python eval/create_demo_data.py --output_dir ./data --num_scenes 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def create_object_image(size: int = 512, shape: str = "ellipse") -> tuple[Image.Image, Image.Image]:
    """Create a synthetic object (colored ellipse) on a background.

    Returns:
        image: RGB image with the object
        mask: L-mode binary mask of the object
    """
    img = Image.new("RGB", (size, size), color=(200, 200, 220))  # light gray bg
    mask = Image.new("L", (size, size), 0)
    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)

    # Random object color and position
    rng = np.random.default_rng()
    color = tuple(rng.integers(50, 200, size=3).tolist())
    cx, cy = size // 2, size // 2
    rx, ry = size // 4, size // 3

    bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
    draw_img.ellipse(bbox, fill=color, outline=(0, 0, 0), width=2)
    draw_mask.ellipse(bbox, fill=255)

    # Add some detail
    detail_color = tuple(min(c + 40, 255) for c in color)
    small_bbox = [cx - rx // 3, cy - ry // 2, cx + rx // 3, cy]
    draw_img.ellipse(small_bbox, fill=detail_color)

    return img, mask


def crop_partial_view(
    image: Image.Image,
    mask: Image.Image,
    visibility: float = 0.5,
) -> tuple[Image.Image, Image.Image]:
    """Crop the image so only `visibility` fraction of the object is visible.

    Simulates a partial view by shifting the crop window.
    """
    w, h = image.size
    mask_np = np.array(mask)
    ys, xs = np.where(mask_np > 128)

    if len(ys) == 0:
        return image.copy(), mask.copy()

    obj_top, obj_bot = ys.min(), ys.max()
    obj_height = obj_bot - obj_top

    # Shift crop to hide (1-visibility) of the object from the bottom
    hidden_pixels = int(obj_height * (1 - visibility))
    crop_top = hidden_pixels

    # Crop and resize back
    cropped_img = image.crop((0, crop_top, w, crop_top + h))
    cropped_mask = mask.crop((0, crop_top, w, crop_top + h))

    # Pad to original size (add black at bottom)
    padded_img = Image.new("RGB", (w, h), (200, 200, 220))
    padded_mask = Image.new("L", (w, h), 0)
    paste_h = min(h, cropped_img.size[1])
    padded_img.paste(cropped_img.crop((0, 0, w, paste_h)), (0, 0))
    padded_mask.paste(cropped_mask.crop((0, 0, w, paste_h)), (0, 0))

    return padded_img, padded_mask


def create_camera_pose(position: list[float], look_at: list[float] | None = None) -> dict:
    """Create a simple camera pose."""
    # Simplified extrinsic (translation only for demo)
    ext = np.eye(4).tolist()
    ext[0][3] = position[0]
    ext[1][3] = position[1]
    ext[2][3] = position[2]

    intr = [
        [256.0, 0.0, 256.0],
        [0.0, 256.0, 256.0],
        [0.0, 0.0, 1.0],
    ]

    return {
        "extrinsic": ext,
        "intrinsic": intr,
        "width": 512,
        "height": 512,
    }


def create_scene(scene_dir: Path, visibility: float = 0.5):
    """Create a single demo scene."""
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "input").mkdir(exist_ok=True)
    (scene_dir / "gt").mkdir(exist_ok=True)
    (scene_dir / "poses").mkdir(exist_ok=True)
    (scene_dir / "masks").mkdir(exist_ok=True)

    # Create full-view object (GT)
    gt_image, gt_mask = create_object_image()

    # Create partial-view (input)
    input_image, input_mask = crop_partial_view(gt_image, gt_mask, visibility=visibility)

    # Save
    gt_image.save(scene_dir / "gt" / "image.png")
    gt_mask.save(scene_dir / "masks" / "gt_mask.png")
    input_image.save(scene_dir / "input" / "image.png")
    input_mask.save(scene_dir / "masks" / "input_mask.png")

    # Camera poses
    input_pose = create_camera_pose([0.0, 0.5, 2.0])
    target_pose = create_camera_pose([0.0, 0.0, 2.0])

    with open(scene_dir / "poses" / "input_pose.json", "w") as f:
        json.dump(input_pose, f, indent=2)
    with open(scene_dir / "poses" / "target_pose.json", "w") as f:
        json.dump(target_pose, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--num_scenes", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create scenes with varying input visibility
    visibilities = np.linspace(0.2, 0.8, args.num_scenes)

    for i, vis in enumerate(visibilities):
        scene_id = f"scene_{i:03d}"
        scene_dir = output_dir / scene_id
        create_scene(scene_dir, visibility=vis)
        print(f"Created {scene_id} (input visibility: {vis:.0%})")

    print(f"\nCreated {args.num_scenes} demo scenes in {output_dir}")
    print("Run evaluation with:")
    print(f"  python eval/evaluate.py --data_root {output_dir} --device cpu")


if __name__ == "__main__":
    main()
