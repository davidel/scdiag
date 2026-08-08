"""Automatic state-dict key alignment between two models.

Given an *old* state-dict (e.g. pretrained checkpoint) and a *new* state-dict
(e.g. target model to initialise), produce a ``{new_key: old_key}`` mapping.

Matching is driven by:
  1. **Shape** — old and new parameter shapes must be identical.
  2. **Name tokens** — keys are split on ``"."`` into tokens, reversed (leaf
     first), and compared from the leaf side.  A longer leaf-side match wins
     over a shorter one, which naturally handles prefix differences (different
     top-level naming) and layer expansion/contraction (extra layers have no
     matching old key).
"""

from collections import defaultdict


class AlignConfig:
  """Configuration for state-dict key alignment.

  Attributes:
      max_prefix_drop: Maximum root-side (leading) token mismatch tolerated.
      min_token_match: Minimum number of consecutive leaf-side tokens that
          must match for a valid match.
  """

  def __init__(self, max_prefix_drop=2, min_token_match=2):
    self.max_prefix_drop = max_prefix_drop
    self.min_token_match = min_token_match


class AlignReport:
  """Result of :func:`align_state_dicts`.

  Attributes:
      mapping: ``{new_key: old_key}`` for successfully matched parameters.
      unmatched_new: New-model keys that could not be matched.
      unused_old: Old-model keys that were never selected.
      divergent: Old key claimed by multiple new keys (1→M).
  """

  def __init__(self):
    self.mapping = {}
    self.unmatched_new = []
    self.unused_old = []
    self.divergent = []

  @property
  def ok(self):
    """True when every new key was matched and every old key was used."""
    return (not self.unmatched_new and not self.unused_old and not self.divergent)


def _tokenize(key):
  return tuple(reversed(key.split(".")))


class _TreeNode:

  def __init__(self):
    self.children = defaultdict(_TreeNode)
    self.keys = []


def _build_shape_tree(old_state):
  tree = defaultdict(_TreeNode)
  for key, tensor in old_state.items():
    shape = tuple(tensor.shape)
    tokens = _tokenize(key)
    node = tree[shape]
    for tok in tokens:
      node = node.children[tok]
    node.keys.append(key)
  return tree


def _walk_tree(shape_tree, shape, tokens, max_prefix_drop):
  matches = []
  if shape in shape_tree:
    root = shape_tree[shape]
    new_len = len(tokens)

    def _collect(node):
      for k in node.keys:
        old_tokens = _tokenize(k)
        old_len = len(old_tokens)
        score = 0
        for a, b in zip(old_tokens, tokens):
          if a == b:
            score += 1
          else:
            break
        shared = min(old_len, new_len)
        root_side_mismatches = sum(
            1 for i in range(score, shared) if old_tokens[i] != tokens[i])
        extra = abs(old_len - new_len)
        total_prefix_drop = root_side_mismatches + extra
        if total_prefix_drop <= max_prefix_drop:
          matches.append((k, score))
      for child in node.children.values():
        _collect(child)

    _collect(root)
  return matches


def align_state_dicts(old_state, new_state, config=None):
  """Align keys between *old_state* (pretrained) and *new_state* (target).

  Args:
      old_state: State-dict of the source/pretrained model.
      new_state: State-dict of the target model to initialise.
      config: Alignment configuration.  Uses defaults when ``None``.

  Returns:
      :class:`AlignReport` with the mapping and diagnostics.
  """
  if config is None:
    config = AlignConfig()

  shape_tree = _build_shape_tree(old_state)
  report = AlignReport()
  old_to_new = defaultdict(list)

  for new_key, tensor in new_state.items():
    new_shape = tuple(tensor.shape)
    new_tokens = _tokenize(new_key)
    candidates = _walk_tree(shape_tree, new_shape, new_tokens, config.max_prefix_drop)
    candidates = [(k, s) for k, s in candidates if s >= config.min_token_match]
    if candidates:
      candidates.sort(key=lambda x: (-x[1], x[0]))
      best_key = candidates[0][0]
      report.mapping[new_key] = best_key
      old_to_new[best_key].append(new_key)
    else:
      report.unmatched_new.append(new_key)

  for old_key, new_keys in old_to_new.items():
    if len(new_keys) > 1:
      report.divergent.append((old_key, new_keys))

  selected_old = set(report.mapping.values())
  report.unused_old = sorted(set(old_state.keys()) - selected_old)

  return report


def create_report(report):
  """Return a human-readable string summarising the alignment result."""
  lines = []

  if not report.unmatched_new and not report.unused_old and not report.divergent:
    n = len(report.mapping)
    lines.append(f"param_align: all {n} new keys matched, {n} old keys used.")
  else:
    if report.unmatched_new:
      lines.append(
          f"param_align: {len(report.unmatched_new)} new keys UNMATCHED"
          " (will be randomly initialised):")
      for k in report.unmatched_new:
        lines.append(f"  UNMATCHED → {k}")

    if report.unused_old:
      lines.append(f"param_align: {len(report.unused_old)} old keys"
                   " UNUSED (will not be copied):")
      for k in report.unused_old:
        lines.append(f"  UNUSED ← {k}")

    if report.divergent:
      lines.append(f"param_align: {len(report.divergent)} old keys claimed"
                   " by multiple new keys:")
      for old_key, new_keys in report.divergent:
        lines.append(f"  1→M ← {old_key} → {new_keys}")

  return "\n".join(lines)
