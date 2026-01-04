"""
Formatting utilities for terminal UI display.

This module provides formatting functions for creating sparklines, colored
blocks, reward breakdowns, and other visual elements in the terminal UI.
"""

from typing import Dict, List, Optional, Union
import numpy as np


def create_sparkline(
    data: List[float],
    width: Optional[int] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> str:
    """
    Create a Unicode sparkline from numeric data.

    Args:
        data: List of numeric values to visualize
        width: Optional width in characters (defaults to len(data))
        min_val: Optional minimum value for scaling (defaults to min(data))
        max_val: Optional maximum value for scaling (defaults to max(data))

    Returns:
        str: Unicode sparkline string

    Examples:
        >>> create_sparkline([1, 2, 3, 4, 5])
        '▁▃▅▆█'
        >>> create_sparkline([5, 4, 3, 2, 1])
        '█▆▅▃▁'
    """
    if not data:
        return ""

    # Sparkline characters from lowest to highest
    chars = "▁▂▃▄▅▆▇█"

    # Handle single value
    if len(data) == 1:
        return chars[len(chars) // 2]

    # Resample data if width is specified and different from data length
    if width and width != len(data):
        indices = np.linspace(0, len(data) - 1, width)
        data = [data[int(i)] for i in indices]

    # Determine range
    data_min = min_val if min_val is not None else min(data)
    data_max = max_val if max_val is not None else max(data)

    # Handle case where all values are the same
    if data_max == data_min:
        return chars[len(chars) // 2] * len(data)

    # Normalize and convert to sparkline
    sparkline = []
    for value in data:
        normalized = (value - data_min) / (data_max - data_min)
        index = int(normalized * (len(chars) - 1))
        index = max(0, min(len(chars) - 1, index))  # Clamp to valid range
        sparkline.append(chars[index])

    return "".join(sparkline)


def format_sequence_visual(
    sequence: List[int],
    cluster_colors: Optional[Dict[int, str]] = None,
    max_length: int = 50,
) -> str:
    """
    Format a sequence of cluster IDs as colored blocks.

    Args:
        sequence: List of cluster IDs
        cluster_colors: Optional mapping of cluster ID to color name
        max_length: Maximum number of blocks to display

    Returns:
        str: Formatted string with colored blocks

    Examples:
        >>> format_sequence_visual([1, 1, 2, 3, 2, 1])
        '██████'  # With colors applied
    """
    if not sequence:
        return ""

    # Truncate if needed
    if len(sequence) > max_length:
        sequence = sequence[:max_length]

    # Use blocks for visualization
    block = "█"

    # If no color mapping provided, use plain blocks
    if not cluster_colors:
        return block * len(sequence)

    # Create colored sequence
    from rich.text import Text
    text = Text()
    for cluster_id in sequence:
        color = cluster_colors.get(cluster_id, "white")
        text.append(block, style=color)

    return text


def format_reward_breakdown(
    reward_components: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    width: int = 40,
) -> str:
    """
    Format reward components as a breakdown table.

    Args:
        reward_components: Dictionary of component names to values
        weights: Optional dictionary of component weights
        width: Width of the breakdown bar

    Returns:
        str: Formatted reward breakdown string
    """
    from rich.table import Table
    from rich.console import Console
    from io import StringIO

    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("Component", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_column("Weight", justify="right", style="yellow")
    table.add_column("Weighted", justify="right", style="bold green")

    total_weighted = 0.0
    for component, value in reward_components.items():
        weight = weights.get(component, 1.0) if weights else 1.0
        weighted_value = value * weight
        total_weighted += weighted_value

        table.add_row(
            component.capitalize(),
            f"{value:.4f}",
            f"{weight:.2f}",
            f"{weighted_value:.4f}",
        )

    # Add total row
    table.add_row(
        "[bold]Total[/bold]",
        "",
        "",
        f"[bold]{total_weighted:.4f}[/bold]",
        style="bold",
    )

    # Render to string
    console = Console(file=StringIO(), width=width, legacy_windows=False)
    console.print(table)
    return console.file.getvalue()


def format_agent_status(
    agent_type: str,
    metrics: Dict[str, Union[float, int, str]],
) -> str:
    """
    Format agent-specific status information.

    Args:
        agent_type: Type of agent ("GHSOM" or "DQN")
        metrics: Dictionary of metric names to values

    Returns:
        str: Formatted status string
    """
    from rich.table import Table
    from rich.console import Console
    from io import StringIO

    table = Table(title=f"{agent_type} Agent", show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="white")

    for metric_name, value in metrics.items():
        # Format value based on type
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        elif isinstance(value, int):
            formatted_value = f"{value:,}"
        else:
            formatted_value = str(value)

        table.add_row(metric_name.replace("_", " ").title(), formatted_value)

    # Render to string
    console = Console(file=StringIO(), width=50, legacy_windows=False)
    console.print(table)
    return console.file.getvalue()


def format_convergence_indicator(
    current_reward: float,
    reward_history: List[float],
    window_size: int = 10,
) -> str:
    """
    Format convergence indicator based on reward trends.

    Args:
        current_reward: Current episode reward
        reward_history: List of historical rewards
        window_size: Window size for trend calculation

    Returns:
        str: Convergence indicator (emoji + trend)
    """
    if len(reward_history) < window_size:
        return "🔄 Warming up..."

    # Calculate recent trend
    recent = reward_history[-window_size:]
    if len(recent) < 2:
        return "🔄 Training..."

    # Simple linear trend
    x = np.arange(len(recent))
    y = np.array(recent)
    slope = np.polyfit(x, y, 1)[0]

    # Determine trend
    if abs(slope) < 0.001:
        return "⚖️  Stable"
    elif slope > 0:
        return "📈 Improving"
    else:
        return "📉 Declining"


def format_metric_with_trend(
    metric_name: str,
    current_value: float,
    history: List[float],
    decimals: int = 4,
) -> str:
    """
    Format a metric with its current value and sparkline trend.

    Args:
        metric_name: Name of the metric
        current_value: Current metric value
        history: Historical values for sparkline
        decimals: Number of decimal places

    Returns:
        str: Formatted metric string with sparkline
    """
    sparkline = create_sparkline(history[-20:] if len(history) > 20 else history)
    return f"{metric_name}: {current_value:.{decimals}f} {sparkline}"


def create_progress_bar(
    current: int,
    total: int,
    width: int = 30,
    show_percentage: bool = True,
) -> str:
    """
    Create a text-based progress bar.

    Args:
        current: Current progress value
        total: Total value
        width: Width of the progress bar in characters
        show_percentage: Whether to show percentage

    Returns:
        str: Formatted progress bar string
    """
    if total == 0:
        percentage = 0.0
    else:
        percentage = (current / total) * 100

    filled_width = int((current / total) * width) if total > 0 else 0
    bar = "█" * filled_width + "░" * (width - filled_width)

    if show_percentage:
        return f"[{bar}] {percentage:.1f}%"
    else:
        return f"[{bar}] {current}/{total}"
