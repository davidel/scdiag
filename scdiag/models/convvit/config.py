"""ConvViT configuration."""


class ConvViTConfig:
  """All hyperparameters for the ConvViT architecture."""

  def __init__(
      self,
      image_size=224,
      patch_size=16,
      in_chans=3,
      depths=None,
      embed_dims=None,
      vit_hidden_dim=768,
      vit_num_heads=12,
      vit_num_layers=12,
      vit_mlp_ratio=4.0,
      drop_rate=0.0,
      drop_path_rate=0.1,
      head_hidden_dim=512,
      num_labels=7,
      id2label=None,
      label2id=None,
  ):
    self.image_size = image_size
    self.patch_size = patch_size
    self.in_chans = in_chans
    self.depths = depths if depths is not None else [2, 3, 8, 3]
    self.embed_dims = embed_dims if embed_dims is not None else [128, 256, 512, 768]
    self.vit_hidden_dim = vit_hidden_dim
    self.vit_num_heads = vit_num_heads
    self.vit_num_layers = vit_num_layers
    self.vit_mlp_ratio = vit_mlp_ratio
    self.drop_rate = drop_rate
    self.drop_path_rate = drop_path_rate
    self.head_hidden_dim = head_hidden_dim
    self.num_labels = num_labels
    self.id2label = id2label if id2label is not None else {}
    self.label2id = label2id if label2id is not None else {}
