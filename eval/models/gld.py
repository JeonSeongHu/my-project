"""GLD (Generative Latent Dynamics) model wrapper for NVS evaluation.

Paper: https://arxiv.org/abs/2405.11730
Repo: https://github.com/apple/ml-gld

Adapt the import paths below to match your local GLD installation.
"""

import torch
import numpy as np

from .base import BaseNVSModel


class GLDModel(BaseNVSModel):
    """Wrapper for GLD novel view synthesis."""

    def __init__(self, **kwargs):
        self.model = None
        self.device = "cuda"
        self.config = None

    def load(self, checkpoint: str | None = None, config: str | None = None,
             device: str = "cuda") -> None:
        self.device = device

        # -----------------------------------------------------------------
        # TODO: Replace with actual GLD imports from your installation.
        #
        # Example (adapt paths to your setup):
        #
        #   import sys
        #   sys.path.insert(0, "/path/to/ml-gld")
        #   from gld.model import GLD
        #   from gld.config import load_config
        #
        #   self.config = load_config(config)
        #   self.model = GLD(self.config)
        #   self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        #   self.model.to(device).eval()
        # -----------------------------------------------------------------

        try:
            import sys
            # Adjust this path to your GLD repo location
            # sys.path.insert(0, "/path/to/ml-gld")
            from gld.model import GLD as GLDNet
            from gld.config import load_config

            self.config = load_config(config) if config else {}
            self.model = GLDNet(self.config)
            if checkpoint:
                state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
                self.model.load_state_dict(state_dict)
            self.model.to(device).eval()
            print("[GLD] Model loaded successfully.")
        except ImportError:
            print("[GLD] WARNING: GLD not installed. Using dummy output.")
            print("      Install GLD and update the import paths in eval/models/gld.py")

    def _pose_to_matrix(self, pose: dict) -> torch.Tensor:
        """Convert pose dict to 4x4 extrinsic matrix."""
        if "extrinsic" in pose:
            return torch.tensor(pose["extrinsic"], dtype=torch.float32, device=self.device)
        return torch.eye(4, device=self.device)

    def _get_intrinsic(self, pose: dict) -> torch.Tensor:
        if "intrinsic" in pose:
            return torch.tensor(pose["intrinsic"], dtype=torch.float32, device=self.device)
        # Default intrinsic (512x512, fx=fy=256)
        return torch.tensor([
            [256.0, 0, 256.0],
            [0, 256.0, 256.0],
            [0, 0, 1],
        ], device=self.device)

    @torch.no_grad()
    def synthesize(
        self,
        input_image: torch.Tensor,
        input_pose: dict,
        target_pose: dict,
    ) -> torch.Tensor:
        """Generate novel view using GLD.

        Args:
            input_image: [3, H, W] in [0, 1]
            input_pose: camera pose for input
            target_pose: camera pose for target

        Returns:
            [3, H, W] generated view in [0, 1]
        """
        if self.model is None:
            # Dummy: return blurred input as placeholder
            print("[GLD] No model loaded, returning dummy output.")
            return _dummy_nvs(input_image)

        img = input_image.unsqueeze(0).to(self.device)  # [1, 3, H, W]
        src_ext = self._pose_to_matrix(input_pose).unsqueeze(0)
        tgt_ext = self._pose_to_matrix(target_pose).unsqueeze(0)
        intrinsic = self._get_intrinsic(input_pose).unsqueeze(0)

        # -----------------------------------------------------------------
        # TODO: Adapt this call to match the actual GLD API.
        #
        # Example:
        #   output = self.model.render(
        #       images=img,
        #       src_cameras=src_ext,
        #       tgt_cameras=tgt_ext,
        #       intrinsics=intrinsic,
        #   )
        #   return output["rgb"].squeeze(0).clamp(0, 1)
        # -----------------------------------------------------------------
        output = self.model.render(
            images=img,
            src_cameras=src_ext,
            tgt_cameras=tgt_ext,
            intrinsics=intrinsic,
        )
        return output["rgb"].squeeze(0).clamp(0, 1)


def _dummy_nvs(image: torch.Tensor) -> torch.Tensor:
    """Placeholder: Gaussian-blur the input as a dummy NVS output."""
    import torch.nn.functional as F
    img = image.unsqueeze(0)
    k = 11
    sigma = 3.0
    coords = torch.arange(k, dtype=torch.float32, device=image.device) - k // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel_1d = gauss.view(1, 1, -1)
    # Separable blur
    blurred = F.conv2d(
        F.pad(img, (k // 2, k // 2, 0, 0), mode="reflect"),
        kernel_1d.unsqueeze(2).expand(3, 1, 1, k),
        groups=3,
    )
    blurred = F.conv2d(
        F.pad(blurred, (0, 0, k // 2, k // 2), mode="reflect"),
        kernel_1d.unsqueeze(3).expand(3, 1, k, 1),
        groups=3,
    )
    return blurred.squeeze(0).clamp(0, 1)
