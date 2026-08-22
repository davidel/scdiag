"""TimmProcessor — bridges timm's data config to the scdiag processor protocol.

timm embeds preprocessing metadata (mean, std, interpolation, crop
percentage) inside each model's ``pretrained_cfg`` dict.  There is no
standalone "processor" object like HuggingFace's ``AutoImageProcessor``.

This class wraps the resolved data config so that:

* ``.image_mean`` / ``.image_std`` satisfy the normalization protocol
  used by ``build_transforms`` and ``build_val_transform``.
* ``__call__(image)`` applies the full model-specific transform pipeline
  (resize, crop, normalize) via ``timm.data.create_transform``.
"""

import timm.data


class TimmProcessor:
  """Thin adapter that exposes timm preprocessing as a processor object."""

  def __init__(self, data_config, image_size):
    """
    Parameters
    ----------
    data_config : dict
      Output of ``timm.data.resolve_data_config()`` — contains
      ``mean``, ``std``, ``input_size``, ``interpolation``,
      ``crop_pct``, ``crop_mode``.
    image_size : int
      Target resolution (pixels) after resize / crop.
    """
    self._mean = data_config["mean"]
    self._std = data_config["std"]
    self._data_config = data_config
    self.image_size = image_size

    # Build a timm-native transform so __call__ respects the model's
    # own interpolation and crop_pct settings.
    cfg = dict(data_config)
    cfg["input_size"] = (3, image_size, image_size)
    self._transform = timm.data.create_transform(**cfg)

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

  def __call__(self, image):
    """Apply the full model-specific transform pipeline.

    Parameters
    ----------
    image : PIL.Image.Image
      Raw input image.

    Returns
    -------
    torch.Tensor
      Shape ``(C, H, W)`` — normalized and resized to ``image_size``.
    """
    return self._transform(image)

  def __repr__(self):
    return (f"TimmProcessor(image_size={self.image_size}, "
            f"mean={self._mean}, std={self._std})")
