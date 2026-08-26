"""FieldSectorDataset — extract a single field from a dict-returning dataset."""


class FieldSectorDataset:
  """Wrap a dict-returning dataset and return only one field's value.

  Parameters
  ----------
  source : Dataset
      A dataset whose ``__getitem__`` returns a ``dict``.
  field : str
      The key to extract from each dict.

  Example
  -------
  >>> ds = FieldSectorDataset(my_dataset, field="image")
  >>> tensor = ds[0]  # returns just the image
  """

  def __init__(self, source, field):
    self._source = source
    self._field = field

  def __len__(self):
    return len(self._source)

  def __getitem__(self, idx):
    return self._source[idx][self._field]
