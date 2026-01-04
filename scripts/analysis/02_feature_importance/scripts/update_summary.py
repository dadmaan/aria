#!/usr/bin/env python3
"""
Update the comprehensive summary report with VIF findings
"""

import pandas as pd
import json
from src.utils.logging.logging_manager import LoggingManager

OUTPUT_DIR = "/home/user/pilot_study/experiments_feature_analysis/feature_analysis_20251113_214037/"

# Initialize logging
logger = LoggingManager(
    name="ghsom_feature_analysis.update_summary",
    log_file=f"{OUTPUT_DIR}logs/update_summary.log",
)

logger.info("Updating comprehensive summary report with VIF findings...")

# Load existing summary
with open(f"{OUTPUT_DIR}reports/eda_summary.json", "r") as f:
    summary_report = json.load(f)

# Load VIF scores
vif_df = pd.read_csv(f"{OUTPUT_DIR}data/vif_scores.csv")

# Update multicollinearity analysis
high_vif = vif_df[vif_df["VIF"] > 10].copy()
infinite_vif = vif_df[vif_df["VIF"] == float("inf")].copy()
finite_vif = vif_df[vif_df["VIF"] != float("inf")].dropna()

summary_report["multicollinearity_analysis"] = {
    "vif_computed": True,
    "total_features_analyzed": len(vif_df),
    "features_with_high_vif": high_vif["feature"].tolist(),
    "features_with_infinite_vif": infinite_vif["feature"].tolist(),
    "vif_stats": {
        "mean_vif": float(finite_vif["VIF"].mean()) if len(finite_vif) > 0 else None,
        "median_vif": (
            float(finite_vif["VIF"].median()) if len(finite_vif) > 0 else None
        ),
        "max_finite_vif": (
            float(finite_vif["VIF"].max()) if len(finite_vif) > 0 else None
        ),
        "min_vif": float(finite_vif["VIF"].min()) if len(finite_vif) > 0 else None,
    },
    "top_vif_features": [
        {
            "feature": row["feature"],
            "VIF": float(row["VIF"]) if row["VIF"] != float("inf") else "inf",
        }
        for _, row in high_vif.head(15).iterrows()
    ],
}

# Save updated summary
with open(f"{OUTPUT_DIR}reports/eda_summary.json", "w") as f:
    json.dump(summary_report, f, indent=2)

logger.info("✓ Summary report updated successfully")

# Print summary
logger.info("\n" + "=" * 80)
logger.info("UPDATED SUMMARY")
logger.info("=" * 80)

logger.info("\nVIF Analysis:")
logger.info("  • Total features analyzed: %d", len(vif_df))
logger.info("  • Features with VIF > 10: %d", len(high_vif))
logger.info("  • Features with infinite VIF: %d", len(infinite_vif))

if len(finite_vif) > 0:
    logger.info("\n  Finite VIF statistics:")
    logger.info("    - Mean VIF: %.2f", finite_vif["VIF"].mean())
    logger.info("    - Median VIF: %.2f", finite_vif["VIF"].median())
    logger.info("    - Max finite VIF: %.2f", finite_vif["VIF"].max())
    logger.info("    - Min VIF: %.2f", finite_vif["VIF"].min())

logger.info("\n  Top 10 features with highest finite VIF:")
top_finite = finite_vif.nlargest(10, "VIF")
for _, row in top_finite.iterrows():
    logger.info("    - %s VIF=%s", f"{row['feature']:50s}", f"{row['VIF']:10.2f}")

if len(infinite_vif) > 0:
    logger.info("\n  Features with perfect multicollinearity (infinite VIF):")
    for feat in infinite_vif["feature"]:
        logger.info("    - %s", feat)

logger.info("\n" + "=" * 80)
