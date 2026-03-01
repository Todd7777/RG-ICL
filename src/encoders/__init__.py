from .base import BaseEncoder, EncoderOutput
from .dinov3 import DINOv3Encoder
from .clip_encoder import CLIPEncoder
from .mae import MAEEncoder

ENCODERS = {
    "dinov3": DINOv3Encoder,
    "clip": CLIPEncoder,
    "mae": MAEEncoder,
}


def get_encoder(name: str, **kwargs) -> BaseEncoder:
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder: {name}. Available: {list(ENCODERS.keys())}")
    return ENCODERS[name](**kwargs)
