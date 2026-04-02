from .base import BaseNVSModel
from .gld import GLDModel
from .seva import SevaModel

MODEL_REGISTRY = {
    "gld": GLDModel,
    "seva": SevaModel,
}


def build_model(name: str, **kwargs) -> BaseNVSModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
