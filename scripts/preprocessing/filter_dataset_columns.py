#!/usr/bin/env python3
"""
CSV Dataset Column Filter Script

Filters CSV datasets by selecting specific columns and outputs the filtered data
to artifacts/features/filtered/{DATASET_NAME}/ with comprehensive logging.

Usage:
    python scripts/filter_dataset_columns.py \
        --input artifacts/features/raw/commu_full/features_numeric.csv \
        --columns "pm_note_count,pm_tempo_bpm,track_id,metadata_index"
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import pandas as pd

# Import LoggingManager from project's logging infrastructure
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.logging.logging_manager import LoggingManager


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Filter CSV dataset by selecting specific columns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s --input data/features.csv --columns "col1,col2,col3"
  
  # With custom output directory
  %(prog)s --input data/features.csv --columns "col1,col2" --output results/
  
  # With debug logging
  %(prog)s --input data/features.csv --columns "col1,col2" --verbose DEBUG
        """,
    )

    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Path to input CSV file"
    )

    parser.add_argument(
        "-c",
        "--columns",
        type=str,
        required=True,
        help="Comma-separated list of column names to keep",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output directory path (default: artifacts/features/filtered/{DATASET_NAME}/)",
    )

    parser.add_argument(
        "-l",
        "--log-file",
        type=str,
        default=None,
        help="Log file path (default: logs/filter_dataset_{timestamp}.log)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity level (default: INFO)",
    )

    return parser.parse_args()


def validate_input_file(file_path: Path, logger: LoggingManager) -> bool:
    """Validate that input file exists and is readable.

    Args:
        file_path: Path to input CSV file
        logger: LoggingManager instance for logging

    Returns:
        True if file is valid, False otherwise
    """
    if not file_path.exists():
        logger.error("Input file does not exist: %s", file_path)
        return False

    if not file_path.is_file():
        logger.error("Input path is not a file: %s", file_path)
        return False

    if file_path.suffix.lower() != ".csv":
        logger.warning("Input file does not have .csv extension: %s", file_path)

    logger.debug("Input file validated: %s", file_path)
    return True


def parse_column_list(columns_str: str, logger: LoggingManager) -> List[str]:
    """Parse comma-separated column names into a list.

    Args:
        columns_str: Comma-separated column names
        logger: LoggingManager instance for logging

    Returns:
        List of column names
    """
    columns = [col.strip() for col in columns_str.split(",") if col.strip()]
    logger.debug("Parsed %d columns: %s", len(columns), columns)
    return columns


def determine_output_path(
    input_path: Path, output_arg: Optional[str], logger: LoggingManager
) -> Path:
    """Determine the output directory path based on input and arguments.

    Args:
        input_path: Path to input CSV file
        output_arg: Optional output directory argument
        logger: LoggingManager instance for logging

    Returns:
        Path to output directory
    """
    if output_arg:
        output_dir = Path(output_arg)
        logger.debug("Using custom output directory: %s", output_dir)
    else:
        # Auto-detect dataset name from input path structure
        # Expected: artifacts/features/raw/{DATASET_NAME}/file.csv
        # Output: artifacts/features/filtered/{DATASET_NAME}/
        try:
            if "raw" in input_path.parts:
                raw_index = input_path.parts.index("raw")
                if raw_index + 1 < len(input_path.parts):
                    dataset_name = input_path.parts[raw_index + 1]
                    output_dir = Path("artifacts/features/filtered") / dataset_name
                    logger.debug("Auto-detected dataset name: %s", dataset_name)
                else:
                    # Fallback: use parent directory name
                    dataset_name = input_path.parent.name
                    output_dir = Path("artifacts/features/filtered") / dataset_name
                    logger.debug(
                        "Using parent directory as dataset name: %s", dataset_name
                    )
            else:
                # Fallback: use parent directory name
                dataset_name = input_path.parent.name
                output_dir = Path("artifacts/features/filtered") / dataset_name
                logger.debug("Using parent directory as dataset name: %s", dataset_name)
        except Exception as e:
            logger.warning("Could not auto-detect dataset name: %s. Using 'default'", e)
            output_dir = Path("artifacts/features/filtered/default")

    logger.info("Output directory: %s", output_dir)
    return output_dir


def load_csv(file_path: Path, logger: LoggingManager) -> Optional[pd.DataFrame]:
    """Load CSV file into a pandas DataFrame.

    Args:
        file_path: Path to CSV file
        logger: LoggingManager instance for logging

    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        logger.info("Loading CSV file: %s", file_path)
        df = pd.read_csv(file_path)
        logger.info(
            "Successfully loaded CSV with %d rows and %d columns",
            len(df),
            len(df.columns),
        )
        logger.debug("Available columns: %s", list(df.columns))
        return df
    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty: %s", file_path)
        return None
    except pd.errors.ParserError as e:
        logger.error("Failed to parse CSV file: %s", e)
        return None
    except Exception as e:
        logger.error("Unexpected error loading CSV: %s", e)
        return None


def validate_columns(
    df: pd.DataFrame, requested_columns: List[str], logger: LoggingManager
) -> bool:
    """Validate that all requested columns exist in the DataFrame.

    Args:
        df: Input DataFrame
        requested_columns: List of requested column names
        logger: LoggingManager instance for logging

    Returns:
        True if all columns exist, False otherwise
    """
    available_columns = set(df.columns)
    missing_columns = [col for col in requested_columns if col not in available_columns]

    if missing_columns:
        logger.error(
            "The following columns do not exist in the dataset: %s", missing_columns
        )
        logger.info("Available columns: %s", list(df.columns))

        # Provide suggestions for similar column names
        for missing_col in missing_columns:
            suggestions = [
                col for col in available_columns if missing_col.lower() in col.lower()
            ]
            if suggestions:
                logger.info("Did you mean one of these? %s", suggestions)

        return False

    logger.debug("All requested columns are present in the dataset")
    return True


def filter_columns(
    df: pd.DataFrame, columns: List[str], logger: LoggingManager
) -> pd.DataFrame:
    """Filter DataFrame to include only specified columns.

    Args:
        df: Input DataFrame
        columns: List of column names to keep
        logger: LoggingManager instance for logging

    Returns:
        Filtered DataFrame
    """
    logger.info("Filtering dataset to %d columns", len(columns))
    filtered_df = df[columns]
    logger.info(
        "Filtered dataset shape: %d rows × %d columns",
        len(filtered_df),
        len(filtered_df.columns),
    )
    logger.debug("Retained columns: %s", list(filtered_df.columns))
    return filtered_df


def save_csv(
    df: pd.DataFrame, output_dir: Path, filename: str, logger: LoggingManager
) -> bool:
    """Save DataFrame to CSV file with atomic write operation.

    Args:
        df: DataFrame to save
        output_dir: Output directory path
        filename: Output filename
        logger: LoggingManager instance for logging

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured output directory exists: %s", output_dir)

        output_path = output_dir / filename
        temp_path = output_dir / f".{filename}.tmp"

        # Write to temporary file first
        logger.debug("Writing to temporary file: %s", temp_path)
        df.to_csv(temp_path, index=False)

        # Atomic move to final destination
        temp_path.rename(output_path)
        logger.info("Successfully saved filtered dataset to: %s", output_path)
        logger.info("Output file size: %d rows × %d columns", len(df), len(df.columns))

        return True

    except PermissionError:
        logger.error("Permission denied when writing to: %s", output_dir)
        return False
    except OSError as e:
        logger.error("OS error when saving file: %s", e)
        return False
    except Exception as e:
        logger.error("Unexpected error saving CSV: %s", e)
        return False


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = args.log_file or f"logs/filter_dataset_{timestamp}.log"

    log_level_map = {
        "DEBUG": 10,  # logging.DEBUG
        "INFO": 20,  # logging.INFO
        "WARNING": 30,  # logging.WARNING
        "ERROR": 40,  # logging.ERROR
    }

    logger = LoggingManager(
        name="csv_filter", level=log_level_map[args.verbose], log_file=log_file
    )

    # Log script start
    logger.info("=" * 60)
    logger.info("CSV Dataset Column Filter Script")
    logger.info("=" * 60)
    logger.info("Input file: %s", args.input)
    logger.info("Requested columns: %s", args.columns)
    logger.info("Log file: %s", log_file)

    # Convert input path to Path object
    input_path = Path(args.input)

    # Validate input file
    if not validate_input_file(input_path, logger):
        logger.error("Input validation failed. Exiting.")
        sys.exit(1)

    # Parse column list
    columns = parse_column_list(args.columns, logger)
    if not columns:
        logger.error("No valid columns specified. Exiting.")
        sys.exit(1)

    # Load CSV
    df = load_csv(input_path, logger)
    if df is None:
        logger.error("Failed to load CSV file. Exiting.")
        sys.exit(1)

    # Validate columns exist
    if not validate_columns(df, columns, logger):
        logger.error("Column validation failed. Exiting.")
        sys.exit(1)

    # Filter columns
    filtered_df = filter_columns(df, columns, logger)

    # Determine output path
    output_dir = determine_output_path(input_path, args.output, logger)

    # Save filtered dataset
    if not save_csv(filtered_df, output_dir, input_path.name, logger):
        logger.error("Failed to save filtered dataset. Exiting.")
        sys.exit(1)

    # Success summary
    logger.info("=" * 60)
    logger.info("Filtering completed successfully!")
    logger.info(
        "Original: %d columns → Filtered: %d columns", len(df.columns), len(columns)
    )
    logger.info("Output: %s", output_dir / input_path.name)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
