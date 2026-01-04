"""LSTM hidden state diagnostic utilities.

This module provides tools for detecting, logging, and tracking
LSTM hidden state issues during training. It is part of Phase 1
of the LSTM Hidden State Infrastructure Fix initiative.

The key problem being addressed:
    All 4 recurrent network implementations (drqn.py, dueling_drqn.py,
    c51_drqn.py, rainbow_drqn.py) silently reset LSTM hidden states to
    zeros when batch size mismatches occur during training, effectively
    negating the purpose of using recurrent networks.

This module provides visibility into:
    - When batch size mismatches occur
    - How often they occur (mismatch rate)
    - Hidden state norm statistics
    - Batch size distribution during training

"""

import logging
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class LSTMStateMetrics:
    """Tracks LSTM hidden state metrics over time.

    This dataclass collects and aggregates metrics about LSTM hidden state
    behavior during training, enabling visibility into batch size mismatches
    and hidden state characteristics.

    Attributes:
        mismatch_count: Number of times batch size mismatch occurred.
        total_forward_calls: Total number of forward passes recorded.
        h_norm_history: Recent hidden state norms (max 1000 samples).
        c_norm_history: Recent cell state norms (max 1000 samples).
        batch_sizes_seen: Distribution of batch sizes encountered.

    Example:
        >>> metrics = LSTMStateMetrics()
        >>> h = torch.randn(1, 4, 128)
        >>> c = torch.randn(1, 4, 128)
        >>> metrics.record_forward(h, c)
        >>> print(f"Calls: {metrics.total_forward_calls}")
        Calls: 1
    """

    # Mismatch tracking
    mismatch_count: int = 0
    total_forward_calls: int = 0

    # State norm history (recent 1000 samples)
    h_norm_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    c_norm_history: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Batch size tracking
    batch_sizes_seen: Dict[int, int] = field(default_factory=dict)

    def record_mismatch(self, expected_batch: int, actual_batch: int) -> None:
        """Record a hidden state batch size mismatch.

        Called when the hidden state batch size doesn't match the input
        batch size, triggering a fallback to zero initialization.

        Args:
            expected_batch: Batch size from the hidden state tensor.
            actual_batch: Batch size from the input observation.
        """
        self.mismatch_count += 1
        self.batch_sizes_seen[actual_batch] = (
            self.batch_sizes_seen.get(actual_batch, 0) + 1
        )

    def record_forward(self, h: torch.Tensor, c: torch.Tensor) -> None:
        """Record hidden state metrics from a forward pass.

        Tracks the L2 norm of hidden and cell states for monitoring
        hidden state behavior over time.

        Args:
            h: Hidden state tensor of shape (num_layers, batch, hidden_size).
            c: Cell state tensor of shape (num_layers, batch, hidden_size).
        """
        self.total_forward_calls += 1

        # Compute L2 norms (detach to avoid affecting gradients)
        with torch.no_grad():
            h_norm = h.norm().item()
            c_norm = c.norm().item()

        self.h_norm_history.append(h_norm)
        self.c_norm_history.append(c_norm)

    @property
    def mismatch_rate(self) -> float:
        """Calculate the rate of batch size mismatches.

        Returns:
            Fraction of forward calls that had batch size mismatches.
            Returns 0.0 if no forward calls have been recorded.
        """
        if self.total_forward_calls == 0:
            return 0.0
        return self.mismatch_count / self.total_forward_calls

    @property
    def avg_h_norm(self) -> float:
        """Average hidden state norm over recent samples.

        Returns:
            Mean L2 norm of hidden states, or 0.0 if no samples.
        """
        if not self.h_norm_history:
            return 0.0
        return sum(self.h_norm_history) / len(self.h_norm_history)

    @property
    def avg_c_norm(self) -> float:
        """Average cell state norm over recent samples.

        Returns:
            Mean L2 norm of cell states, or 0.0 if no samples.
        """
        if not self.c_norm_history:
            return 0.0
        return sum(self.c_norm_history) / len(self.c_norm_history)

    def to_dict(self) -> Dict:
        """Export metrics as dictionary for logging.

        Useful for exporting to W&B or other logging systems.

        Returns:
            Dictionary containing all key metrics.
        """
        return {
            "mismatch_count": self.mismatch_count,
            "mismatch_rate": self.mismatch_rate,
            "total_forward_calls": self.total_forward_calls,
            "avg_h_norm": self.avg_h_norm,
            "avg_c_norm": self.avg_c_norm,
            "batch_sizes_seen": dict(self.batch_sizes_seen),
        }

    def reset(self) -> None:
        """Reset all metrics to initial state.

        Useful for clearing metrics between experiments or tests.
        """
        self.mismatch_count = 0
        self.total_forward_calls = 0
        self.h_norm_history.clear()
        self.c_norm_history.clear()
        self.batch_sizes_seen.clear()


# Global metrics instance (can be replaced per-network)
_global_metrics: Optional[LSTMStateMetrics] = None

# Per-network metrics instances
_network_metrics: Dict[str, LSTMStateMetrics] = {}


def get_metrics(network_name: Optional[str] = None) -> LSTMStateMetrics:
    """Get or create LSTM metrics instance.

    Args:
        network_name: Optional network identifier. If provided, returns
            metrics specific to that network. If None, returns global metrics.

    Returns:
        LSTMStateMetrics instance for the specified network or global.

    Example:
        >>> metrics = get_metrics("DRQN")
        >>> metrics.record_forward(h, c)
    """
    global _global_metrics, _network_metrics

    if network_name is None:
        if _global_metrics is None:
            _global_metrics = LSTMStateMetrics()
        return _global_metrics

    if network_name not in _network_metrics:
        _network_metrics[network_name] = LSTMStateMetrics()
    return _network_metrics[network_name]


def reset_metrics(network_name: Optional[str] = None) -> None:
    """Reset metrics to initial state.

    Args:
        network_name: Optional network identifier. If provided, resets only
            that network's metrics. If None, resets global metrics and all
            per-network metrics.

    Example:
        >>> reset_metrics()  # Reset all
        >>> reset_metrics("DRQN")  # Reset only DRQN metrics
    """
    global _global_metrics, _network_metrics

    if network_name is None:
        _global_metrics = LSTMStateMetrics()
        _network_metrics.clear()
    elif network_name in _network_metrics:
        _network_metrics[network_name] = LSTMStateMetrics()


def get_all_network_metrics() -> Dict[str, LSTMStateMetrics]:
    """Get all per-network metrics instances.

    Returns:
        Dictionary mapping network names to their metrics.
    """
    return dict(_network_metrics)


def warn_state_mismatch(
    network_name: str,
    expected_batch: int,
    actual_batch: int,
    log_level: str = "warning",
) -> None:
    """Emit warning when hidden state batch size mismatch occurs.

    This function is called when the hidden state's batch dimension
    doesn't match the input batch size, indicating that the hidden
    state will be reset to zeros (losing temporal context).

    Args:
        network_name: Name of the network (e.g., "DRQN", "C51DRQN").
        expected_batch: Expected batch size from hidden state.
        actual_batch: Actual batch size from input.
        log_level: Logging level - "warning", "debug", or "once".
            - "warning": Always log as warning
            - "debug": Log as debug (less verbose)
            - "once": Use warnings.warn (shown once per location)

    Example:
        >>> warn_state_mismatch("DRQN", 1, 32, log_level="debug")
    """
    message = (
        f"[{network_name}] LSTM hidden state batch size mismatch: "
        f"state has {expected_batch}, input has {actual_batch}. "
        f"Falling back to zero state. This indicates improper hidden "
        f"state handling during training and may degrade performance."
    )

    if log_level == "warning":
        logger.warning(message)
    elif log_level == "debug":
        logger.debug(message)
    elif log_level == "once":
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    else:
        # Default to debug for unknown levels
        logger.debug(message)

    # Record in metrics (both global and per-network)
    get_metrics().record_mismatch(expected_batch, actual_batch)
    get_metrics(network_name).record_mismatch(expected_batch, actual_batch)


def log_state_metrics(
    h: torch.Tensor,
    c: torch.Tensor,
    network_name: str = "LSTM",
    log_every_n: int = 1000,
) -> None:
    """Log hidden state metrics periodically.

    Records hidden and cell state norms for tracking, and optionally
    logs summary statistics at regular intervals.

    Args:
        h: Hidden state tensor of shape (num_layers, batch, hidden_size).
        c: Cell state tensor of shape (num_layers, batch, hidden_size).
        network_name: Network name for logging context.
        log_every_n: Log summary metrics every N forward calls.

    Example:
        >>> h = torch.randn(1, 4, 128)
        >>> c = torch.randn(1, 4, 128)
        >>> log_state_metrics(h, c, "DRQN", log_every_n=100)
    """
    # Record in both global and per-network metrics
    global_metrics = get_metrics()
    network_metrics = get_metrics(network_name)

    global_metrics.record_forward(h, c)
    network_metrics.record_forward(h, c)

    # Log summary periodically
    if network_metrics.total_forward_calls % log_every_n == 0:
        logger.info(
            f"[{network_name}] LSTM State Metrics @ {network_metrics.total_forward_calls} calls: "
            f"mismatch_rate={network_metrics.mismatch_rate:.4f}, "
            f"avg_h_norm={network_metrics.avg_h_norm:.4f}, "
            f"avg_c_norm={network_metrics.avg_c_norm:.4f}"
        )


def generate_diagnostic_report(network_name: Optional[str] = None) -> str:
    """Generate a formatted diagnostic report.

    Creates a human-readable report of LSTM hidden state diagnostics,
    suitable for logging or saving to a file.

    Args:
        network_name: Optional network to report on. If None, reports
            on all networks and global metrics.

    Returns:
        Formatted string report.

    Example:
        >>> report = generate_diagnostic_report("DRQN")
        >>> print(report)
    """
    lines = [
        "=" * 60,
        "LSTM Hidden State Diagnostic Report",
        "=" * 60,
        "",
    ]

    if network_name is not None:
        # Single network report
        metrics = get_metrics(network_name)
        lines.extend(_format_metrics_section(network_name, metrics))
    else:
        # Global metrics
        global_metrics = get_metrics()
        lines.extend(_format_metrics_section("Global", global_metrics))
        lines.append("")

        # Per-network metrics
        for name, metrics in _network_metrics.items():
            lines.extend(_format_metrics_section(name, metrics))
            lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def _format_metrics_section(name: str, metrics: LSTMStateMetrics) -> list:
    """Format a metrics section for the diagnostic report.

    Args:
        name: Section name.
        metrics: Metrics instance to format.

    Returns:
        List of formatted lines.
    """
    lines = [
        f"Network: {name}",
        "-" * 40,
        f"  Total Forward Calls: {metrics.total_forward_calls:,}",
        f"  Mismatch Count: {metrics.mismatch_count:,}",
        f"  Mismatch Rate: {metrics.mismatch_rate:.4%}",
        f"  Avg Hidden State Norm: {metrics.avg_h_norm:.4f}",
        f"  Avg Cell State Norm: {metrics.avg_c_norm:.4f}",
    ]

    if metrics.batch_sizes_seen:
        lines.append("  Batch Size Distribution:")
        for batch_size, count in sorted(metrics.batch_sizes_seen.items()):
            lines.append(f"    {batch_size}: {count:,}")

    return lines
