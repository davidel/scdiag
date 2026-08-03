"""ConvViT image processor — mirrors the HF AutoImageProcessor interface.

Applies the same Resize → CenterCrop → ToImage → ToDtype → Normalize
pipeline that HuggingFace uses for ConvNeXtV2 models.
"""

import torch
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

# Default ImageNet normalization used by ConvNeXtV2 pre-training.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ConvViTProcessor:
  """Image processor for ConvViT.

    Accepts PIL images (single or list) and returns normalised tensors
    ready for the model's ``forward()`` method.
    """

  def __init__(
      self,
      image_size=224,
      mean=IMAGENET_MEAN,
      std=IMAGENET_STD,
  ):
    self.image_size = image_size
    self.mean = mean
    self.std = std

    self._transform = v2.Compose([
        v2.Resize(
            image_size,
            interpolation=InterpolationMode.BICUBIC,
        ),
        v2.CenterCrop(image_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  def __call__(self, images):
    """Process one or more PIL images into a ``pixel_values`` tensor.

        Parameters
        ----------
        images : PIL.Image or list[PIL.Image]
            Raw input image(s).

        Returns
        -------
        torch.Tensor
            Shape ``[B, 3, H, W]`` (always 4-D, even for a single image).
        """
    if isinstance(images, list):
      return torch.stack([self._transform(img) for img in images])
    # Single image → unsqueeze to add batch dim.
    return self._transform(images).unsqueeze(0)

  @property
  def size(self):
    """Return processor resolution as a dict (HF compatibility)."""
    return {"height": self.image_size, "width": self.image_size}

  def __repr__(self):
    return (f"ConvViTProcessor(image_size={self.image_size}, "
            f"mean={self.mean}, std={self.std})")
