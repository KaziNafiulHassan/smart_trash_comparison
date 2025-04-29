#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to parse simulation log files into Pandas DataFrames.
This script reads structured log files from the simulation runs and
combines them into a single DataFrame for analysis.
"""

import os
import json
import glob
import logging
import argparse
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_log_file(filepath):
    """
    Parse a single log file into a pandas DataFrame.

    Args:
        filepath (str): Path to the log file

    Returns:
        pandas.DataFrame: DataFrame containing the parsed log data
    """
    logger.info(f"Parsing log file: {filepath}")

    try:
        # Extract backend type from filename
        filename = os.path.basename(filepath)
        backend_type = filename.split('_')[0]

        # Determine file format based on extension
        _, ext = os.path.splitext(filepath)

        if ext.lower() == '.csv':
            # Parse CSV format
            df = pd.read_csv(filepath)

        elif ext.lower() == '.json':
            # Parse JSON format (assuming one object per line)
            with open(filepath, 'r') as f:
                data = []
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            data.append(record)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON line: {line[:50]}...")

            if data:
                df = pd.DataFrame(data)
            else:
                logger.warning(f"No valid JSON records found in {filepath}")
                return pd.DataFrame()

        elif ext.lower() == '.log':
            # Parse structured log format
            # Assuming each log entry is a JSON object on a single line
            # or a CSV-like format with consistent columns
            with open(filepath, 'r') as f:
                content = f.readlines()

            # Try to determine if it's JSON or CSV-like
            data = []
            for line in content:
                line = line.strip()
                if not line:
                    continue

                # Check if it's a JSON line
                if line.startswith('{') and line.endswith('}'):
                    try:
                        record = json.loads(line)
                        data.append(record)
                    except json.JSONDecodeError:
                        # Not a valid JSON line, skip
                        pass
                elif ' - ' in line and ',' in line:
                    # Might be a CSV-like format with timestamp
                    # Try to extract structured data
                    parts = line.split(' - ', 3)
                    if len(parts) >= 3:
                        # Extract timestamp and message parts
                        timestamp = parts[0]
                        level = parts[1]
                        message = parts[-1]

                        # Parse the message part if it contains key-value pairs
                        record = {'timestamp': timestamp, 'log_level': level}

                        # Look for key=value patterns
                        kvs = re.findall(r'(\w+)=([^,]+)', message)
                        for k, v in kvs:
                            record[k] = v.strip()

                        if len(record) > 2:  # More than just timestamp and log_level
                            data.append(record)

            if data:
                df = pd.DataFrame(data)
            else:
                # If no structured data found, return empty DataFrame
                logger.warning(f"No structured data found in {filepath}")
                return pd.DataFrame()

        else:
            logger.warning(f"Unsupported file extension: {ext}")
            return pd.DataFrame()

        # Add backend_type if not already present
        if 'backend_type' not in df.columns:
            df['backend_type'] = backend_type

        logger.info(f"Successfully parsed {len(df)} records from {filepath}")
        return df

    except Exception as e:
        logger.error(f"Error parsing {filepath}: {str(e)}")
        return pd.DataFrame()

def clean_dataframe(df):
    """
    Clean and prepare the DataFrame for analysis.

    Args:
        df (pandas.DataFrame): DataFrame to clean

    Returns:
        pandas.DataFrame: Cleaned DataFrame
    """
    if df.empty:
        return df

    # Make a copy to avoid modifying the original
    df = df.copy()

    # Convert timestamp to datetime if it exists and isn't already
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_dtype(df['timestamp']):
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except:
            logger.warning("Failed to convert timestamp column to datetime")

    # Convert numeric columns
    numeric_columns = [
        'cv_confidence',
        'processing_latency_ms',
        'user_error_prob_before',
        'user_error_prob_after'
    ]

    for col in numeric_columns:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                logger.warning(f"Failed to convert {col} to numeric")

    # Convert boolean columns
    bool_columns = ['is_correct']
    for col in bool_columns:
        if col in df.columns and not pd.api.types.is_bool_dtype(df[col]):
            try:
                # Handle various representations of boolean values
                df[col] = df[col].map({'true': True, 'false': False, 'True': True, 'False': False, 1: True, 0: False})
            except:
                logger.warning(f"Failed to convert {col} to boolean")

    return df

def main():
    """Main function to parse log files and save the combined DataFrame."""
    parser = argparse.ArgumentParser(description='Parse simulation log files into a pandas DataFrame')

    parser.add_argument(
        '--log_dir',
        type=str,
        default='../logs/',
        help='Path to the directory containing log files (default: ../logs/)'
    )

    parser.add_argument(
        '--log_pattern',
        type=str,
        default='*.log',
        help='Glob pattern to select specific log files (default: *.log)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='combined_simulation_data.pkl',
        help='Output file path for the combined DataFrame (default: combined_simulation_data.pkl)'
    )

    args = parser.parse_args()

    # Find all log files matching the pattern
    log_pattern = os.path.join(args.log_dir, args.log_pattern)
    log_files = glob.glob(log_pattern)

    if not log_files:
        logger.error(f"No log files found matching pattern: {log_pattern}")
        return

    logger.info(f"Found {len(log_files)} log files matching pattern: {log_pattern}")

    # Parse each log file and collect the DataFrames
    dataframes = []
    for log_file in log_files:
        df = parse_log_file(log_file)
        if not df.empty:
            dataframes.append(df)

    if not dataframes:
        logger.error("No data extracted from log files")
        return

    # Combine all DataFrames
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined DataFrame has {len(combined_df)} rows and {len(combined_df.columns)} columns")

    # Clean the combined DataFrame
    combined_df = clean_dataframe(combined_df)

    # Save the combined DataFrame
    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save based on file extension
        _, ext = os.path.splitext(args.output)

        if ext.lower() == '.pkl':
            combined_df.to_pickle(args.output)
        elif ext.lower() == '.csv':
            combined_df.to_csv(args.output, index=False)
        elif ext.lower() == '.feather':
            combined_df.to_feather(args.output)
        else:
            # Default to pickle
            output_file = args.output if ext else f"{args.output}.pkl"
            combined_df.to_pickle(output_file)

        logger.info(f"Saved combined DataFrame to {args.output}")

    except Exception as e:
        logger.error(f"Error saving combined DataFrame: {str(e)}")

if __name__ == "__main__":
    import re  # Import here to avoid unused import warning
    main()
