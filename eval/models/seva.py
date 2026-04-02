"""Seva model wrapper for NVS evaluation.

Paper: https://arxiv.org/abs/2410.01680
Repo: https://github.com/Stability-AI/seva

Adapt the import paths below to match your local Seva installation.
"""

import torch
import numpy as np

from .base import BaseNVSModel


class SevaModel(BaseNVSModel):
    """Wrapper for Seva novel view synthesis."""

    def __init__(self, **kwargs):
        self.model = None
        self.device = "cuda"

    def load(self, checkpoint: str | None = None, config: str | None = None,
             device: str = "cuda") -> None:
        self.device = device

        # -----------------------------------------------------------------
        # TODO: Replace with actual Seva imports from your installation.
        #
        # Example (adapt paths):
        #
        #   import sys
        #   sys.path.insert(0, "/path/to/seva")
        #   from seva.model import Seva
        #   from seva.config import SevaConfig
        #
        #   cfg = SevaConfig.from_file(config) if config else SevaConfig()
        #   self.model = Seva(cfg)
        #   self.model.load_state_dict(torch.load(checkpoint))
        #   self.model.to(device).eval()
        # -----------------------------------------------------------------

        try:
            from seva.model import Seva
            from seva.config import SevaConfig

            cfg = SevaConfig.from_file(config) if config else SevaConfig()
            self.model = Seva(cfg)
            if checkpoint:
                state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
                self.model.load_state_dict(state_dict)
            self.model.to(device).eval()
            print("[Seva] Model loaded successfully.")
        except ImportError:
            print("[Seva] WARNING: Seva not installed. Using dummy output.")
            print("       Install Seva and update the import paths in eval/models/seva.py")

    def _build_camera_params(self, input_pose: dict, target_pose: dict) -> dict:
        """Build Seva-compatible camera parameters."""
        def to_tensor(pose, key, default):
            if key in pose:
                return torch.tensor(pose[key], dtype=torch.float32, device=self.device)
            return default

        src_ext = to_tensor(input_pose, "extrinsic", torch.eye(4, device=self.device))
        tgt_ext = to_tensor(target_pose, "extrinsic", torch.eye(4, device=self.device))
        intrinsic = to_tensor(input_pose, "intrinsic",
                              torch.tensor([[256, 0, 256], [0, 256, 256], [0, 0, 1]],
                                           dtype=torch.float32, device=self.device))

        return {
            "src_extrinsic": src_ext.unsqueeze(0),
            "tgt_extrinsic": tgt_ext.unsqueeze(0),
            "intrinsic": intrinsic.unsqueeze(0),
        }

    @torch.no_grad()
    def synthesize(
        self,
        input_image: torch.Tensor,
        input_pose: dict,
        target_pose: dict,
    ) -> torch.Tensor:
        """Generate novel view using Seva.

        Args:
            input_image: [3, H, W] in [0, 1]
            input_pose: camera pose for input
            target_pose: camera pose for target

        Returns:
            [3, H, W] generated view in [0, 1]
        """
        if self.model is None:
            print("[Seva] No model loaded, returning dummy output.")
            from .gld import _dummy_nvs
            return _dummy_nvs(input_image)

        img = input_image.unsqueeze(0).to(self.device)  # [1, 3, H, W]
        camera_params = self._build_camera_params(input_pose, target_pose)

        # -----------------------------------------------------------------
        # TODO: Adapt this call to match the actual Seva API.
        #
        # Example:
        #   output = self.model.generate(
        #       images=img,
        #       cameras=camera_params,
        #       num_steps=50,
        #       cfg_scale=3.0,
        #   )
        #   return output.squeeze(0).clamp(0, 1)
        # -----------------------------------------------------------------
        output = self.model.generate(
            images=img,
            cameras=camera_params,
            num_steps=50,
            cfg_scale=3.0,
        )
        return output.squeeze(0).clamp(0, 1)
