"""Reference baselines from torch-harmonics examples.

`natten` is an optional dep for Transformer/Segformer; if missing, a stub
raises ImportError on instantiation.
"""


class _MissingNatten:
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "This baseline requires `natten`; install it to use Transformer/Segformer."
        )


try:
    from baselines.transformer import Transformer
except ImportError:
    Transformer = _MissingNatten
try:
    from baselines.segformer import Segformer
except ImportError:
    Segformer = _MissingNatten
from baselines.unet import UNet

__all__ = ["Segformer", "Transformer", "UNet"]
