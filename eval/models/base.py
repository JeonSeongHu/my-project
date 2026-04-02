"""Base interface for NVS models."""

from abc import ABC, abstractmethod

import torch


class BaseNVSModel(ABC):
    """Abstract base class for Novel View Synthesis models.

    All NVS models must implement `synthesize()` which takes an input image
    and camera poses, and returns a generated novel view.
    """

    @abstractmethod
    def load(self, checkpoint: str | None = None, config: str | None = None,
             device: str = "cuda") -> None:
        """Load model weights."""

    @abstractmethod
    def synthesize(
        self,
        input_image: torch.Tensor,
        input_pose: dict,
        target_pose: dict,
    ) -> torch.Tensor:
        """Generate a novel view.

        Args:
            input_image: [3, H, W] input observation in [0, 1]
            input_pose: camera pose dict for input view
            target_pose: camera pose dict for target view

        Returns:
            generated: [3, H, W] synthesized novel view in [0, 1]
        """
