"""Lightweight gradient inspector for PyTorch models.

Usage::

    monitor = GradMonitor(model, log_every=50)

    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        monitor.step(global_step)
        optimizer.step()
        optimizer.zero_grad()

The monitor walks ``model.named_parameters()`` on each call to ``step()``
and computes per-parameter gradient statistics (norm, mean, max, sparsity)
and parameter norms.  Reports are logged via ``logging.info()`` every
``log_every`` steps.

When ``detect_nan=True`` the monitor checks every step (not just log
steps) for NaN / Inf in gradient tensors and logs a critical warning
immediately if any are found.
"""

import logging


def _find_common_root(strings):
  """Return the longest dotted prefix shared by all *strings*.

  Splits each string on ``'.'`` and compares component-by-component,
  so the result is always a clean dotted path (never cuts mid-component).

  Returns ``""`` when no common root exists.
  """
  if not strings:
    return ""
  split = [s.split(".") for s in strings]
  common = []
  for components in zip(*split):
    if len(set(components)) == 1:
      common.append(components[0])
    else:
      break
  return ".".join(common)


def _strip_common_prefix(rows):
  """Strip the common dotted prefix from parameter names.

  Returns ``(prefix, stripped_rows)``.  Each row is a
  ``(name, stats_dict, status)`` tuple.  When no meaningful prefix
  exists the original rows are returned unchanged.
  """
  prefix = _find_common_root([r[0] for r in rows])
  if not prefix:
    return "", rows
  pfx = prefix + "."
  return pfx, [(name[len(pfx):], s, st) for name, s, st in rows]


class GradMonitor:
  """Architecture-agnostic gradient inspector.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose parameters will be inspected.
    log_every : int
        Compute and log gradient stats every N training steps.
    detect_nan : bool
        If *True*, every call to ``step()`` checks all gradient tensors
        for NaN / Inf (slight overhead).
    stall_window : int
        Number of consecutive log-steps a layer's gradient norm must
        stay below ``norm_floor`` before it is flagged as stalled.
    norm_floor : float
        Gradient-norm threshold below which a layer is considered
        inactive.
    norm_ceiling : float
        Gradient-norm threshold above which a layer is considered
        exploding.
    top_k : int
        Number of healthy (non-anomalous) params to show in the report,
        sorted by gradient norm descending.
    imbalance_factor : float
        A parameter whose gradient norm exceeds this factor times the
        median gradient norm is flagged as IMBALANCED.  Defaults to 100.
    gpr_ceiling : float
        Gradient-to-parameter ratio threshold above which a parameter
        is flagged as GPR (G/P Ratio).  A value > 1.0 means the
        gradient update is larger than the weight itself.  Defaults to
        1.0.
    """

  def __init__(
      self,
      model,
      log_every=50,
      detect_nan=False,
      stall_window=50,
      norm_floor=1e-8,
      norm_ceiling=100.0,
      top_k=5,
      imbalance_factor=100.0,
      gpr_ceiling=1.0,
  ):
    self._model = model
    self._log_every = log_every
    self._detect_nan = detect_nan
    self._stall_window = stall_window
    self._norm_floor = norm_floor
    self._norm_ceiling = norm_ceiling
    self._top_k = top_k
    self._imbalance_factor = imbalance_factor
    self._gpr_ceiling = gpr_ceiling

    self._consecutive_low = {}

  def step(self, global_step):
    """Advance the monitor by one training step.

        If ``detect_nan`` is enabled, checks for NaN/Inf every call.
        Computes stats and logs a report every ``log_every`` steps.
        """
    if self._detect_nan:
      self._check_nan()

    if self._log_every > 0 and (global_step % self._log_every == 0):
      self._snapshot(global_step)

  def _check_nan(self):
    for name, p in self._model.named_parameters():
      if p.grad is None:
        continue
      if p.grad.isnan().any() or p.grad.isinf().any():
        logging.critical("GradMonitor: NaN/Inf detected in gradient of '%s'", name)

  def _snapshot(self, global_step):
    stats = {}
    anomalies = []

    for name, p in self._model.named_parameters():
      if p.requires_grad:
        entry = {}
        entry["param_norm"] = float(p.data.norm())

        if p.grad is None:
          entry["grad_norm"] = 0.0
          entry["grad_mean"] = 0.0
          entry["grad_max"] = 0.0
          entry["grad_sparsity"] = 1.0
          entry["grad_param_ratio"] = 0.0
        else:
          g = p.grad.data
          g_norm = float(g.norm())
          entry["grad_norm"] = g_norm
          entry["grad_mean"] = float(g.float().mean())
          entry["grad_max"] = float(g.abs().max())
          entry["grad_sparsity"] = float((g.abs() < 1e-7).float().mean())
          entry["grad_param_ratio"] = g_norm / (entry["param_norm"] + 1e-12)

        is_low = entry["grad_norm"] < self._norm_floor
        prev = self._consecutive_low.get(name, 0)
        self._consecutive_low[name] = prev + 1 if is_low else 0
        if self._consecutive_low[name] >= self._stall_window:
          anomalies.append(f"STALLED({name}, {self._consecutive_low[name]} steps)")

        if entry["grad_norm"] > self._norm_ceiling:
          anomalies.append(f"EXPLODING({name}, norm={entry['grad_norm']:.2e})")

        if entry["grad_param_ratio"] > self._gpr_ceiling:
          anomalies.append(
              f"GPR({name}, ratio={entry['grad_param_ratio']:.2e})")

        stats[name] = entry

    norms = [v["grad_norm"] for v in stats.values() if v["grad_norm"] > 0]
    if norms:
      norms_sorted = sorted(norms)
      median = norms_sorted[len(norms_sorted) // 2]
      for name, v in stats.items():
        if (v["grad_norm"] > 0 and v["grad_norm"] > median * self._imbalance_factor):
          anomalies.append(f"IMBALANCED({name})")

    self._log(global_step, stats, anomalies)

  def _log(self, global_step, stats, anomalies):
    norms = [v["grad_norm"] for v in stats.values()]
    ratios = [
        v["grad_param_ratio"] for v in stats.values() if v["grad_param_ratio"] > 0
    ]

    lines = [(f"[Step {global_step}] Gradient Report:"
              f" {len(stats)} params"
              f" | grad_norm: mean={sum(norms)/len(norms) if norms else 0:.2e}"
              f" max={max(norms) if norms else 0:.2e}"
              f" min={min(norms) if norms else 0:.2e}"
              f" | grad/param: mean={sum(ratios)/len(ratios) if ratios else 0:.2e}")]

    name_status = {}
    for a in anomalies:
      for tag, short in [
          ("STALLED(", "STL"),
          ("EXPLODING(", "OVF"),
          ("IMBALANCED(", "IMB"),
          ("GPR(", "GPR"),
      ]:
        if tag in a:
          start = a.index(tag) + len(tag)
          end = a.index(")", start)
          pname = a[start:end].split(",")[0]
          prev = name_status.get(pname)
          name_status[pname] = f"{prev}|{short}" if prev else short

    SEVERITY = {"STL": 0, "OVF": 1, "IMB": 2, "GPR": 3}

    def sort_key(kv):
      name, s = kv
      sev = SEVERITY.get(name_status.get(name, ""), 99)
      return (sev, -s["grad_norm"])

    ranked = sorted(stats.items(), key=sort_key)
    rows = []
    for name, s in ranked:
      if len(rows) >= self._top_k + len(name_status):
        break
      if s["grad_norm"] > 0 or name in name_status:
        rows.append((name, s, name_status.get(name, "OK")))

    if rows:
      prefix, rows = _strip_common_prefix(rows)
      if prefix:
        lines.append(f"  Prefix (stripped): {prefix}")

      nw = max(len(name) for name, _, _ in rows)
      nw = max(nw, len("Name"))
      hdr = (f"  {'Name':<{nw}} {'g_norm':>10} {'p_norm':>10}"
             f" {'g/p':>10} {'g_max':>10} {'sparse':>7} {'status':<12}")
      sep = "  " + "-" * len(hdr)
      lines.append(hdr)
      lines.append(sep)
      for name, s, status in rows:
        lines.append("  " + self._format_row(name, s, status, nw))

    logging.info("\n".join(lines))

  def _format_row(self, name, s, status, name_width):
    return (f"{name:<{name_width}}"
            f" {s['grad_norm']:>10.2e}"
            f" {s['param_norm']:>10.2e}"
            f" {s['grad_param_ratio']:>10.2e}"
            f" {s['grad_max']:>10.2e}"
            f" {s['grad_sparsity']:>6.1%}"
            f" {status:<12}")
