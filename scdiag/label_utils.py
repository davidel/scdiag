"""Helpers for resolving model class labels consistently."""


def get_label(id2label, class_id):
  """Return a label for an integer class ID.

  Hugging Face configurations usually use string keys, while custom models
  often use integer keys. Support both without mutating the model config.

  Args:
    id2label: Mapping from class IDs to display labels.
    class_id: Numeric class ID to resolve.

  Returns:
    The configured display label.

  Raises:
    KeyError: If the class ID is not present in either supported key form.
  """
  if class_id in id2label:
    return id2label[class_id]
  string_id = str(class_id)
  if string_id in id2label:
    return id2label[string_id]
  raise KeyError(f"No label configured for class ID {class_id!r}")
