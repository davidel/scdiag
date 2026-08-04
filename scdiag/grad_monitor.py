"""Lightweight gradient inspector for PyTorch models.

Usage::

    monitor = GradMonitor(model, log_every=50)

    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        monitor.step(global_step)
        optimizer.step()
        optimizer.zero_grad()

        if monitor.report():
            logger.info(monitor.report())

The monitor walks ``model.named_parameters()`` on each call to ``step()``
and computes per-parameter gradient statistics (norm, mean, max, sparsity)
and parameter norms.  When ``global_step`` falls on a log interval, the
statistics are frozen into an internal snapshot that can be retrieved via
``report()``.

When ``detect_nan=True`` the monitor checks every step (not just log
steps) for NaN / Inf in gradient tensors and logs a critical warning
immediately if any are found.
"""

import logging

import torch


class GradMonitor:
  """Architecture-agnostic gradient inspector.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose parameters will be inspected.
    log_every : int
        Interval (in steps) between report snapshots.
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
    """

  def __init__(self,
               model,
               log_every=50,
               detect_nan=False,
               stall_window=50,
               norm_floor=1e-8,
               norm_ceiling=100.0,
               top_k=5):
    self._model = model
    self._log_every = log_every
    self._detect_nan = detect_nan
    self._stall_window = stall_window
    self._norm_floor = norm_floor
    self._norm_ceiling = norm_ceiling
    self._top_k = top_k

    self._consecutive_low = {}
    self._last_report = None

  def step(self, global_step):
    """Advance the monitor by one training step.

        * If ``detect_nan`` is enabled, gradient tensors are scanned for
          NaN / Inf (logs a CRITICAL message on first detection).
        * If ``global_step`` falls on the log interval, a full snapshot
          is computed and stored internally (accessible via ``report()``).
        """
    if self._detect_nan:
      self._check_nan()

    if self._log_every > 0 and (global_step % self._log_every == 0):
      self._snapshot(global_step)

  def report(self):
    """Return the most recent report string, or *None*."""
    return self._last_report

  def _check_nan(self):
    for name, p in self._model.named_parameters():
      if p.grad is None:
        continue
      if p.grad.isnan().any() or p.grad.isinf().any():
        logging.critical(
            "GradMonitor: NaN/Inf detected in gradient of '%s'",
            name,
        )

  def _snapshot(self, global_step):
    """Collect stats for every parameter and build a report."""
    stats = {}
    anomalies = []

    for name, p in self._model.named_parameters():
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

      stats[name] = entry

    norms = [v["grad_norm"] for v in stats.values() if v["grad_norm"] > 0]
    if norms:
      norms_sorted = sorted(norms)
      median = norms_sorted[len(norms_sorted) // 2]
      for name, v in stats.items():
        if v["grad_norm"] > 0 and v["grad_norm"] > median * 100:
          anomalies.append(f"IMBALANCED({name})")

    self._last_report = self._format(global_step, stats, anomalies)

  def _format(self, global_step, stats, anomalies):
    """Build a human-readable report string."""
    norms = [v["grad_norm"] for v in stats.values()]
    ratios = [
        v["grad_param_ratio"] for v in stats.values() if v["grad_param_ratio"] > 0
    ]

    lines = [
        f"[Step {global_step}] Gradient Report:"
        f" {len(stats)} params"
        f" | grad_norm: mean={sum(norms)/len(norms) if norms else 0:.2e}"
        f" max={max(norms) if norms else 0:.2e}"
        f" min={min(norms) if norms else 0:.2e}"
        f" | grad/param: mean={sum(ratios)/len(ratios) if ratios else 0:.2e}"
    ]

    anomalous = set()
    for a in anomalies:
      # Extract param name from STALLED(name, ...), EXPLODING(name, ...),
      # IMBALANCED(name).
      for tag in ("STALLED(", "EXPLODING(", "IMBALANCED("):
        if tag in a:
          start = a.index(tag) + len(tag)
          end = a.index(")", start)
          anomalous.add(a[start:end].split(",")[0])

    if anomalies:
      lines.append("  Warnings: " + " | ".join(anomalies))

    # Show anomalous params first, then top-k by grad_norm.
    shown = set()
    rows = []

    for name in anomalous:
      if name in stats:
        rows.append(self._format_row(name, stats[name]))
        shown.add(name)

    ranked = sorted(stats.items(), key=lambda kv: kv[1]["grad_norm"], reverse=True)
    for name, s in ranked:
      if len(rows) >= self._top_k + len(anomalous):
        break
      if name not in shown and s["grad_norm"] > 0:
        rows.append(self._format_row(name, s))
        shown.add(name)

    if rows:
      lines.append(
          f"  {'Name':<45} {'g_norm':>10} {'p_norm':>10} {'g/p':>10} {'g_max':>10} {'sparse':>7}"
      )
      lines.append("  " + "-" * 102)
      for r in rows:
        lines.append("  " + r)

    return "\n".join(lines)

  def _format_row(self, name, s):
    return (f"{name:<45}"
            f" {s['grad_norm']:>10.2e}"
            f" {s['param_norm']:>10.2e}"
            f" {s['grad_param_ratio']:>10.2e}"
            f" {s['grad_max']:>10.2e}"
            f" {s['grad_sparsity']:>6.1%}")
