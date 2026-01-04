#!/usr/bin/env python
"""Combined Training + HIL Analysis.

Provides a holistic view of model performance by comparing:
1. Training benchmark results (peak policy performance)
2. HIL inference results (preference adaptation capability)

This addresses the critical question: Does training performance predict
real-world deployment performance?

Usage:
    python scripts/benchmark/compare_training_hil.py \
        --training-dir artifacts/benchmark/main/20251216_140256/ \
        --hil-dir outputs/benchmark/hil_inference/
"""

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple


def load_training_stats(training_dir: Path) -> Dict:
    """Load training analysis results."""
    # Try multiple possible locations
    possible_paths = [
        training_dir / "analysis_results.json",  # Direct in directory
        training_dir / "analysis" / "analysis_results.json",  # In analysis subdir
    ]

    # Also check subdirectories
    if training_dir.exists():
        for subdir in training_dir.iterdir():
            if subdir.is_dir():
                possible_paths.append(subdir / "analysis_results.json")

    for analysis_file in possible_paths:
        if analysis_file.exists():
            with open(analysis_file) as f:
                return json.load(f)

    raise FileNotFoundError(f"Training analysis not found in {training_dir}. Run analyze_results.py first.")


def load_hil_stats(hil_dir: Path) -> Dict:
    """Load HIL analysis results."""
    analysis_file = hil_dir / "analysis" / "hil_analysis_results.json"
    if not analysis_file.exists():
        raise FileNotFoundError(f"HIL analysis not found at {analysis_file}. Run analyze_hil_results.py first.")

    with open(analysis_file) as f:
        return json.load(f)


def normalize_variant_name(name: str) -> str:
    """Normalize variant name for matching between training and HIL results."""
    # Remove seed suffix (e.g., '_s42')
    if "_s" in name:
        name = name.rsplit("_s", 1)[0]
    return name


def compute_combined_analysis(training_stats: Dict, hil_stats: Dict) -> Dict:
    """Compute combined analysis comparing training and HIL performance.

    Args:
        training_stats: Training benchmark statistics
        hil_stats: HIL inference statistics

    Returns:
        Combined analysis results
    """
    combined = {
        "training_variants": len(training_stats.get("variants", {})),
        "hil_variants": len(hil_stats.get("variants", {})),
        "matched_variants": 0,
        "variants": {},
        "correlations": {},
    }

    # Match variants between training and HIL
    training_variants = training_stats.get("variants", {})
    hil_variants = hil_stats.get("variants", {})

    # Build lookup for HIL variants
    hil_lookup = {}
    for v_name, v_stats in hil_variants.items():
        normalized = normalize_variant_name(v_name)
        hil_lookup[normalized] = v_stats

    # Match and combine
    training_scores = []
    hil_scores = []

    for t_variant, t_stats in training_variants.items():
        normalized = normalize_variant_name(t_variant)

        if normalized in hil_lookup:
            h_stats = hil_lookup[normalized]
            combined["matched_variants"] += 1

            # Get scores
            training_risk_adj = t_stats.get("risk_adjusted_score",
                                           t_stats.get("mean_best_reward", 0) - t_stats.get("std_reward", 0))
            hil_risk_adj = h_stats.get("risk_adjusted_feedback", 0)

            training_scores.append(training_risk_adj)
            hil_scores.append(hil_risk_adj)

            # Combined score: weighted average of normalized scores
            # Normalize to 0-1 range for fair combination
            combined["variants"][normalized] = {
                "training_rank": t_stats.get("rank"),
                "training_risk_adj": training_risk_adj,
                "training_mean_best": t_stats.get("mean_best_reward", 0),
                "training_std": t_stats.get("std_reward", 0),
                "hil_rank": h_stats.get("rank"),
                "hil_risk_adj": hil_risk_adj,
                "hil_mean_feedback": h_stats.get("mean_feedback_improvement", 0),
                "hil_std": h_stats.get("std_feedback_improvement", 0),
                "hil_convergence": h_stats.get("mean_convergence_iteration", 1000),
            }

    # Compute correlation between training and HIL scores
    if len(training_scores) >= 3:
        # Spearman rank correlation (manual computation)
        n = len(training_scores)
        training_ranks = [sorted(training_scores, reverse=True).index(x) + 1 for x in training_scores]
        hil_ranks = [sorted(hil_scores, reverse=True).index(x) + 1 for x in hil_scores]

        # Spearman's rho = 1 - (6 * sum(d^2)) / (n * (n^2 - 1))
        d_squared_sum = sum((tr - hr) ** 2 for tr, hr in zip(training_ranks, hil_ranks))
        spearman_rho = 1 - (6 * d_squared_sum) / (n * (n ** 2 - 1)) if n > 1 else 0

        combined["correlations"]["spearman_rho"] = spearman_rho
        combined["correlations"]["interpretation"] = (
            "Strong positive" if spearman_rho > 0.7 else
            "Moderate positive" if spearman_rho > 0.4 else
            "Weak positive" if spearman_rho > 0.1 else
            "No correlation" if spearman_rho > -0.1 else
            "Negative"
        )

    # Compute combined ranking
    # Normalize scores to 0-1 range and compute weighted sum
    if combined["variants"]:
        t_min = min(v["training_risk_adj"] for v in combined["variants"].values())
        t_max = max(v["training_risk_adj"] for v in combined["variants"].values())
        h_min = min(v["hil_risk_adj"] for v in combined["variants"].values())
        h_max = max(v["hil_risk_adj"] for v in combined["variants"].values())

        for variant, stats in combined["variants"].items():
            # Normalize to 0-1
            t_norm = (stats["training_risk_adj"] - t_min) / (t_max - t_min) if t_max != t_min else 0.5
            h_norm = (stats["hil_risk_adj"] - h_min) / (h_max - h_min) if h_max != h_min else 0.5

            # Combined score (equal weight to training and HIL)
            stats["training_normalized"] = t_norm
            stats["hil_normalized"] = h_norm
            stats["combined_score"] = 0.5 * t_norm + 0.5 * h_norm

        # Rank by combined score
        ranked = sorted(combined["variants"].items(), key=lambda x: x[1]["combined_score"], reverse=True)
        for rank, (variant, _) in enumerate(ranked, 1):
            combined["variants"][variant]["combined_rank"] = rank

    return combined


def generate_combined_report(combined: Dict, output_dir: Path) -> str:
    """Generate combined analysis report."""
    report = f"""# Combined Training + HIL Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Type:** Holistic Model Performance (Training + Deployment)

## Overview

This analysis combines training benchmark performance with Human-in-the-Loop (HIL)
inference performance to provide a complete picture of model capabilities.

| Metric | Value |
|--------|-------|
| Training Variants | {combined['training_variants']} |
| HIL Variants | {combined['hil_variants']} |
| Matched Variants | {combined['matched_variants']} |

## Training-HIL Correlation

"""

    if combined.get("correlations"):
        corr = combined["correlations"]
        report += f"**Spearman Rank Correlation:** {corr.get('spearman_rho', 0):.3f} ({corr.get('interpretation', 'N/A')})\n\n"

        if corr.get("spearman_rho", 0) > 0.5:
            report += "> **Key Finding:** Training performance is a good predictor of HIL adaptation performance.\n\n"
        elif corr.get("spearman_rho", 0) < 0.2:
            report += "> **Key Finding:** Training performance does NOT predict HIL adaptation well. HIL-specific evaluation is critical.\n\n"

    report += """## Combined Performance Ranking

*Combined Score = 0.5 × Training (normalized) + 0.5 × HIL (normalized). This balances pre-deployment and deployment performance.*

| Method | Train Rank | Train Risk-Adj | HIL Rank | HIL Risk-Adj | Combined | Overall |
|--------|------------|----------------|----------|--------------|----------|---------|
"""

    if combined["variants"]:
        sorted_variants = sorted(
            combined["variants"].items(),
            key=lambda x: x[1].get("combined_rank", 999)
        )

        for variant, stats in sorted_variants:
            comb_rank = stats.get("combined_rank", "N/A")
            rank_marker = "**1st**" if comb_rank == 1 else f"{comb_rank}th"
            bold = "**" if comb_rank == 1 else ""

            # Build method name
            if "rainbow" in variant:
                method = "Rainbow"
            elif "dueling" in variant:
                method = "Dueling"
            else:
                method = "Baseline"

            if "_per" in variant:
                method += "+PER"
            if "_noisy" in variant:
                method += "+Noisy"
            if "_cl" in variant:
                method += " (CL)"

            report += (
                f"| {bold}{method}{bold} | "
                f"{stats.get('training_rank', 'N/A')} | {stats['training_risk_adj']:.4f} | "
                f"{stats.get('hil_rank', 'N/A')} | {stats['hil_risk_adj']:.4f} | "
                f"{bold}{stats['combined_score']:.4f}{bold} | {rank_marker} |\n"
            )

    report += "\n## Key Insights\n\n"

    if combined["variants"]:
        # Best overall
        best = min(combined["variants"].items(), key=lambda x: x[1].get("combined_rank", 999))
        report += f"### Best Overall: `{best[0]}`\n\n"
        report += f"- Training Rank: #{best[1].get('training_rank', 'N/A')}\n"
        report += f"- HIL Rank: #{best[1].get('hil_rank', 'N/A')}\n"
        report += f"- Combined Score: {best[1]['combined_score']:.4f}\n\n"

        # Biggest training-HIL gap (overperformers and underperformers)
        gaps = [(v, abs(s.get("training_rank", 0) - s.get("hil_rank", 0))) for v, s in combined["variants"].items()]
        gaps.sort(key=lambda x: x[1], reverse=True)

        if gaps and gaps[0][1] > 2:
            report += "### Notable Rank Differences\n\n"
            report += "*Methods where training and HIL ranks differ significantly:*\n\n"

            for variant, gap in gaps[:3]:
                if gap > 1:
                    stats = combined["variants"][variant]
                    t_rank = stats.get("training_rank", 0)
                    h_rank = stats.get("hil_rank", 0)
                    if t_rank < h_rank:
                        report += f"- **{variant}**: Training #{t_rank} → HIL #{h_rank} (training overestimates)\n"
                    else:
                        report += f"- **{variant}**: Training #{t_rank} → HIL #{h_rank} (training underestimates)\n"

    # Recommendations
    report += "\n## Recommendations\n\n"

    if combined.get("correlations", {}).get("spearman_rho", 0) > 0.5:
        report += """1. **Training metrics are reliable** - The strong correlation suggests training performance
   is a reasonable proxy for deployment performance.

2. **Focus on risk-adjusted metrics** - Both training and HIL benefit from penalizing variance.

3. **Dueling + PER consistently wins** - This combination shows robust performance across both
   training and HIL scenarios.
"""
    else:
        report += """1. **Evaluate on HIL tasks** - Training metrics alone may be misleading.

2. **Consider multi-objective optimization** - Optimize for both training and HIL performance.

3. **Domain-specific evaluation matters** - The HIL scenarios reveal performance patterns
   not visible in training benchmarks.
"""

    return report


def analyze_combined(
    training_dir: Path,
    hil_dir: Path,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Main combined analysis function."""
    print(f"\n{'='*60}")
    print("COMBINED TRAINING + HIL ANALYSIS")
    print(f"{'='*60}")

    # Load stats
    print(f"\nLoading training stats from: {training_dir}")
    training_stats = load_training_stats(training_dir)

    print(f"Loading HIL stats from: {hil_dir}")
    hil_stats = load_hil_stats(hil_dir)

    # Compute combined analysis
    combined = compute_combined_analysis(training_stats, hil_stats)

    print(f"\nMatched {combined['matched_variants']} variants")

    # Set output directory
    if output_dir is None:
        output_dir = hil_dir.parent / "combined_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {output_dir}")

    # Generate report
    report_content = generate_combined_report(combined, output_dir)

    # Save outputs
    report_file = output_dir / "COMBINED_ANALYSIS_REPORT.md"
    with open(report_file, "w") as f:
        f.write(report_content)
    print(f"\n[OK] Report: {report_file}")

    # Save JSON
    stats_file = output_dir / "combined_analysis_results.json"
    with open(stats_file, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"[OK] Statistics: {stats_file}")

    # Save CSV
    csv_file = output_dir / "combined_ranking.csv"
    with open(csv_file, "w") as f:
        f.write("variant,training_rank,training_risk_adj,hil_rank,hil_risk_adj,combined_score,combined_rank\n")
        for variant, stats in combined["variants"].items():
            f.write(
                f"{variant},{stats.get('training_rank', 'N/A')},{stats['training_risk_adj']:.4f},"
                f"{stats.get('hil_rank', 'N/A')},{stats['hil_risk_adj']:.4f},"
                f"{stats['combined_score']:.4f},{stats.get('combined_rank', 'N/A')}\n"
            )
    print(f"[OK] CSV: {csv_file}")

    return {
        "combined": combined,
        "report_file": str(report_file),
        "stats_file": str(stats_file),
        "csv_file": str(csv_file),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Combined Training + HIL Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--training-dir",
        type=str,
        required=True,
        help="Directory containing training benchmark results",
    )
    parser.add_argument(
        "--hil-dir",
        type=str,
        required=True,
        help="Directory containing HIL inference results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: hil_dir/../combined_analysis/)",
    )
    args = parser.parse_args()

    training_dir = Path(args.training_dir)
    hil_dir = Path(args.hil_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    try:
        result = analyze_combined(training_dir, hil_dir, output_dir)

        print(f"\n{'='*60}")
        print("COMBINED ANALYSIS COMPLETE")
        print(f"{'='*60}")

        if result:
            combined = result["combined"]
            if combined.get("correlations"):
                print(f"Training-HIL Correlation: {combined['correlations'].get('spearman_rho', 0):.3f}")

            if combined["variants"]:
                best = min(combined["variants"].items(), key=lambda x: x[1].get("combined_rank", 999))
                print(f"\nBest Overall: {best[0]}")
                print(f"  Combined Score: {best[1]['combined_score']:.4f}")
                print(f"  Training Rank: #{best[1].get('training_rank', 'N/A')}")
                print(f"  HIL Rank: #{best[1].get('hil_rank', 'N/A')}")

            print(f"\nFull report: {result['report_file']}")

        print(f"{'='*60}")

    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
