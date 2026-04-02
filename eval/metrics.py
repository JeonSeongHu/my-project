"""Metrics for NVS Object Completeness Evaluation.

Metrics:
1. Standard NVS: PSNR, SSIM, LPIPS (full image)
2. Object-region NVS: PSNR, SSIM, LPIPS (masked to object region only)
3. Object Completeness Score (OCS): how completely the object is reconstructed
4. Segmentation IoU: overlap between predicted and GT object masks
"""

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Results for a single scene."""
    scene_id: str
    model_name: str

    # Full-image metrics
    psnr: float = 0.0
    ssim: float = 0.0
    lpips: float = 0.0

    # Object-region metrics (computed only on the object mask area)
    obj_psnr: float = 0.0
    obj_ssim: float = 0.0
    obj_lpips: float = 0.0

    # Completeness metrics
    object_completeness_score: float = 0.0  # OCS: ratio of reconstructed object
    mask_iou: float = 0.0                    # IoU between pred and GT object mask
    input_visibility: float = 0.0            # how much was visible in input


class PSNRMetric:
    """Peak Signal-to-Noise Ratio."""

    @staticmethod
    def compute(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> float:
        """
        Args:
            pred: [3, H, W] in [0, 1]
            gt: [3, H, W] in [0, 1]
            mask: [1, H, W] binary mask (optional, compute only on masked region)
        """
        if mask is not None:
            # Expand mask to 3 channels
            mask = mask.expand_as(pred)
            n_pixels = mask.sum().item()
            if n_pixels == 0:
                return 0.0
            mse = ((pred - gt) ** 2 * mask).sum().item() / n_pixels
        else:
            mse = ((pred - gt) ** 2).mean().item()

        if mse == 0:
            return float("inf")
        return 10 * np.log10(1.0 / mse)


class SSIMMetric:
    """Structural Similarity Index (simplified per-channel)."""

    @staticmethod
    def compute(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> float:
        """Compute SSIM using sliding window approach.

        Args:
            pred: [3, H, W] in [0, 1]
            gt: [3, H, W] in [0, 1]
            mask: [1, H, W] binary mask (optional)
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        window_size = 11

        # Create Gaussian window
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) * g.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)

        pred_4d = pred.unsqueeze(0)  # [1, 3, H, W]
        gt_4d = gt.unsqueeze(0)

        mu_pred = F.conv2d(pred_4d, window, padding=window_size // 2, groups=3)
        mu_gt = F.conv2d(gt_4d, window, padding=window_size // 2, groups=3)

        mu_pred_sq = mu_pred ** 2
        mu_gt_sq = mu_gt ** 2
        mu_pred_gt = mu_pred * mu_gt

        sigma_pred_sq = F.conv2d(pred_4d ** 2, window, padding=window_size // 2, groups=3) - mu_pred_sq
        sigma_gt_sq = F.conv2d(gt_4d ** 2, window, padding=window_size // 2, groups=3) - mu_gt_sq
        sigma_pred_gt = F.conv2d(pred_4d * gt_4d, window, padding=window_size // 2, groups=3) - mu_pred_gt

        ssim_map = ((2 * mu_pred_gt + C1) * (2 * sigma_pred_gt + C2)) / \
                   ((mu_pred_sq + mu_gt_sq + C1) * (sigma_pred_sq + sigma_gt_sq + C2))

        ssim_map = ssim_map.squeeze(0)  # [3, H, W]

        if mask is not None:
            mask_3c = mask.expand_as(ssim_map)
            n_pixels = mask_3c.sum().item()
            if n_pixels == 0:
                return 0.0
            return (ssim_map * mask_3c).sum().item() / n_pixels
        return ssim_map.mean().item()


class LPIPSMetric:
    """Learned Perceptual Image Patch Similarity."""

    def __init__(self, device: str = "cuda"):
        try:
            import lpips
            self.model = lpips.LPIPS(net="alex").to(device).eval()
        except ImportError:
            print("WARNING: lpips not installed. Run: pip install lpips")
            self.model = None
        self.device = device

    def compute(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> float:
        """
        Args:
            pred: [3, H, W] in [0, 1]
            gt: [3, H, W] in [0, 1]
            mask: [1, H, W] binary mask (optional, applied before LPIPS)
        """
        if self.model is None:
            return -1.0

        # LPIPS expects [-1, 1]
        pred_scaled = pred.unsqueeze(0).to(self.device) * 2 - 1
        gt_scaled = gt.unsqueeze(0).to(self.device) * 2 - 1

        if mask is not None:
            mask_4d = mask.unsqueeze(0).to(self.device)
            pred_scaled = pred_scaled * mask_4d
            gt_scaled = gt_scaled * mask_4d

        with torch.no_grad():
            return self.model(pred_scaled, gt_scaled).item()


class ObjectCompletenessMetric:
    """Measure how completely an object is reconstructed in the novel view.

    Uses a segmentation model (SAM) to detect the object in both the
    generated view and the GT view, then compares coverage.

    Object Completeness Score (OCS):
        OCS = Area(object in pred) ∩ Area(object in GT) / Area(object in GT)

    This tells us what fraction of the full object the model successfully
    reconstructed, especially the parts that were NOT visible in the input.
    """

    def __init__(self, method: str = "sam", device: str = "cuda",
                 sam_checkpoint: str | None = None, sam_model_type: str = "vit_h"):
        self.method = method
        self.device = device
        self.sam_predictor = None

        if method == "sam" and sam_checkpoint:
            self._init_sam(sam_checkpoint, sam_model_type)

    def _init_sam(self, checkpoint: str, model_type: str):
        """Initialize SAM for automatic object segmentation."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(self.device)
            self.sam_predictor = SamPredictor(sam)
        except ImportError:
            print("WARNING: segment_anything not installed.")

    def segment_object(self, image: torch.Tensor, hint_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Segment the main object in the image.

        Args:
            image: [3, H, W] in [0, 1]
            hint_mask: [1, H, W] optional hint mask (e.g., from GT)

        Returns:
            mask: [1, H, W] binary mask of detected object
        """
        if self.sam_predictor is not None:
            return self._segment_with_sam(image, hint_mask)
        # Fallback: simple saliency-based segmentation
        return self._segment_saliency(image)

    def _segment_with_sam(self, image: torch.Tensor, hint_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Use SAM to segment the object."""
        img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        self.sam_predictor.set_image(img_np)

        if hint_mask is not None:
            # Use center of hint mask as point prompt
            mask_np = hint_mask.squeeze(0).cpu().numpy()
            ys, xs = np.where(mask_np > 0.5)
            if len(ys) > 0:
                cy, cx = int(ys.mean()), int(xs.mean())
                point_coords = np.array([[cx, cy]])
                point_labels = np.array([1])
                masks, scores, _ = self.sam_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
                best_idx = scores.argmax()
                mask = torch.from_numpy(masks[best_idx]).float().unsqueeze(0)
                return mask.to(image.device)

        # No hint: use automatic mask generation (largest object)
        from segment_anything import SamAutomaticMaskGenerator
        # Fallback to center point
        h, w = image.shape[1], image.shape[2]
        point_coords = np.array([[w // 2, h // 2]])
        point_labels = np.array([1])
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        best_idx = scores.argmax()
        mask = torch.from_numpy(masks[best_idx]).float().unsqueeze(0)
        return mask.to(image.device)

    def _segment_saliency(self, image: torch.Tensor) -> torch.Tensor:
        """Fallback: simple saliency-based object detection.

        Uses grayscale variance and Otsu-like thresholding.
        """
        gray = image.mean(dim=0, keepdim=True)  # [1, H, W]

        # Simple edge detection as saliency proxy
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=image.device).view(1, 1, 3, 3)

        gray_4d = gray.unsqueeze(0)
        edges_x = F.conv2d(gray_4d, sobel_x, padding=1)
        edges_y = F.conv2d(gray_4d, sobel_y, padding=1)
        saliency = (edges_x ** 2 + edges_y ** 2).sqrt().squeeze(0)

        # Gaussian blur for smoother regions
        k = 15
        sigma = 5.0
        coords = torch.arange(k, dtype=torch.float32, device=image.device) - k // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        saliency_4d = saliency.unsqueeze(0)
        blurred = F.conv2d(saliency_4d, kernel, padding=k // 2)
        blurred = blurred.squeeze(0)

        # Threshold
        threshold = blurred.mean() + blurred.std()
        mask = (blurred > threshold).float()
        return mask

    def compute_completeness(
        self,
        pred_image: torch.Tensor,
        gt_image: torch.Tensor,
        gt_mask: torch.Tensor | None = None,
    ) -> tuple[float, float, torch.Tensor, torch.Tensor]:
        """Compute Object Completeness Score.

        Args:
            pred_image: [3, H, W] generated novel view
            gt_image: [3, H, W] ground truth novel view
            gt_mask: [1, H, W] GT object mask (if available)

        Returns:
            ocs: Object Completeness Score (0~1)
            iou: Mask IoU between pred and GT object regions
            pred_mask: [1, H, W] detected object mask in pred
            gt_mask_out: [1, H, W] object mask in GT (given or detected)
        """
        # Get GT mask
        if gt_mask is None:
            gt_mask = self.segment_object(gt_image)
        gt_mask_binary = (gt_mask > 0.5).float()

        # Get predicted object mask
        pred_mask = self.segment_object(pred_image, hint_mask=gt_mask_binary)
        pred_mask_binary = (pred_mask > 0.5).float()

        # OCS = |pred ∩ GT| / |GT|
        intersection = (pred_mask_binary * gt_mask_binary).sum().item()
        gt_area = gt_mask_binary.sum().item()

        if gt_area == 0:
            ocs = 1.0
        else:
            ocs = intersection / gt_area

        # IoU = |pred ∩ GT| / |pred ∪ GT|
        union = ((pred_mask_binary + gt_mask_binary) > 0).float().sum().item()
        iou = intersection / union if union > 0 else 1.0

        return ocs, iou, pred_mask_binary, gt_mask_binary


class MetricSuite:
    """Combined metric computation for NVS completeness evaluation."""

    def __init__(self, device: str = "cuda",
                 seg_method: str = "sam",
                 sam_checkpoint: str | None = None,
                 sam_model_type: str = "vit_h"):
        self.psnr = PSNRMetric()
        self.ssim = SSIMMetric()
        self.lpips = LPIPSMetric(device=device)
        self.completeness = ObjectCompletenessMetric(
            method=seg_method, device=device,
            sam_checkpoint=sam_checkpoint, sam_model_type=sam_model_type,
        )
        self.device = device

    def evaluate(
        self,
        pred_image: torch.Tensor,
        gt_image: torch.Tensor,
        gt_mask: torch.Tensor | None,
        scene_id: str,
        model_name: str,
        input_visibility: float,
    ) -> MetricResult:
        """Compute all metrics for a single prediction.

        Args:
            pred_image: [3, H, W] in [0, 1]
            gt_image: [3, H, W] in [0, 1]
            gt_mask: [1, H, W] object mask on GT view
            scene_id: scene identifier
            model_name: name of the NVS model
            input_visibility: fraction of object visible in input

        Returns:
            MetricResult with all computed metrics
        """
        pred = pred_image.to(self.device)
        gt = gt_image.to(self.device)
        mask = gt_mask.to(self.device) if gt_mask is not None else None

        # Full-image metrics
        psnr_val = self.psnr.compute(pred, gt)
        ssim_val = self.ssim.compute(pred, gt)
        lpips_val = self.lpips.compute(pred, gt)

        # Object-region metrics
        obj_psnr = self.psnr.compute(pred, gt, mask=mask) if mask is not None else -1.0
        obj_ssim = self.ssim.compute(pred, gt, mask=mask) if mask is not None else -1.0
        obj_lpips = self.lpips.compute(pred, gt, mask=mask) if mask is not None else -1.0

        # Completeness
        ocs, iou, _, _ = self.completeness.compute_completeness(pred, gt, gt_mask=mask)

        return MetricResult(
            scene_id=scene_id,
            model_name=model_name,
            psnr=psnr_val,
            ssim=ssim_val,
            lpips=lpips_val,
            obj_psnr=obj_psnr,
            obj_ssim=obj_ssim,
            obj_lpips=obj_lpips,
            object_completeness_score=ocs,
            mask_iou=iou,
            input_visibility=input_visibility,
        )
