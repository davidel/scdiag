"""UVito image processor — mirrors the HF AutoImageProcessor interface.

Applies the same Resize → CenterCrop → ToImage → ToDtype → Normalize
pipeline that HuggingFace uses for ConvNeXtV2 models.
"""

import torch
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

# Default ImageNet normalization used by SMP encoder weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.228, 0.227)


class UVitoProcessor:
  """Stateless image pre-processor for UVito.

  Mirrors the interface of a HuggingFace ``AutoImageProcessor``: the
  caller passes ``processor(image)`` and receives a ``[C, H, W]``
  ``float32`` tensor ready for the model.
  """

  def __init__(
      self,
      image_size=384,
      mean=IMAGENET_MEAN,
      std=IMAGENET_STD,
  ):
    self.image_size = image_size
    self._mean = mean
    self._std = std

    # Build a torchvision v2 transform pipeline.
    self._transform = v2.Compose([
        v2.Resize(
            image_size,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        v2.CenterCrop(image_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

  def __call__(self, image, **kwargs):
    return self._transform(image)

  @property
  def image_mean(self):
    """HF-compatible normalization mean."""
    return list(self._mean)

  @property
  def image_std(self):
    """HF-compatible normalization std."""
    return list(self._std)

  @property
  def size(self):
    """Return processor resolution as a dict (HF compatibility)."""
    return {"height": self.image_size, "width": self.image_size}

  def __repr__(self):
    return (f"UVitoProcessor(image_size={self.image_size}, "
            f"mean={self._mean}, std={self._std})")
