#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to analyze simulation data and generate comparison plots.

This script loads combined simulation data from a pickle file (or other formats),
analyzes the data, and generates various plots to compare the performance of
different backends.
"""

import os
import logging
import argparse
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_file):
    """
    Load the combined simulation data from the specified file.

    Args:
        data_file (str): Path to the data file (pickle, csv, or feather)

    Returns:
        pandas.DataFrame: DataFrame with the combined simulation data
    """
    try:
        file_ext = os.path.splitext(data_file)[1].lower()

        if file_ext == '.pkl':
            df = pd.read_pickle(data_file)
        elif file_ext == '.csv':
            df = pd.read_csv(data_file)
        elif file_ext == '.feather':
            df = pd.read_feather(data_file)
        else:
            logger.error(f"Unsupported file extension: {file_ext}")
            return None

        logger.info(f"Loaded data from {data_file}: {len(df)} rows, {len(df.columns)} columns")

        # Convert timestamp to datetime if it exists and isn't already
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    except Exception as e:
        logger.error(f"Error loading data from {data_file}: {str(e)}")
        return None

def plot_overall_accuracy(df, output_dir):
    """
    Calculate and plot overall accuracy by backend type.

    Args:
        df (pandas.DataFrame): DataFrame with simulation data
        output_dir (str): Directory to save the plot
    """
    # Calculate mean is_correct grouped by backend_type
    accuracy_by_backend = df.groupby('backend_type')['is_correct'].mean().sort_values(ascending=False)

    # Print results
    logger.info("Overall Accuracy by Backend Type:")
    for backend, accuracy in accuracy_by_backend.items():
        logger.info(f"  {backend}: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Create bar plot
    plt.figure(figsize=(10, 6))
    ax = accuracy_by_backend.plot(kind='bar', color=sns.color_palette("viridis", len(accuracy_by_backend)))

    # Add labels and title
    plt.xlabel('Backend Type')
    plt.ylabel('Accuracy')
    plt.title('Overall Accuracy by Backend Type')
    plt.ylim(0, 1.0)  # Set y-axis from 0 to 1

    # Add value labels on top of bars
    for i, v in enumerate(accuracy_by_backend):
        ax.text(i, v + 0.02, f"{v:.2%}", ha='center')

    # Save the plot
    output_file = os.path.join(output_dir, 'overall_accuracy.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    logger.info(f"Saved overall accuracy plot to {output_file}")

    plt.close()

def plot_latency(df, output_dir):
    """
    Create box plots of processing latency by backend type.

    Args:
        df (pandas.DataFrame): DataFrame with simulation data
        output_dir (str): Directory to save the plot
    """
    # Check if processing_latency_ms column exists
    if 'processing_latency_ms' not in df.columns:
        logger.warning("No processing_latency_ms column found in data, skipping latency plot")
        return

    # Calculate median and mean latency by backend type
    latency_stats = df.groupby('backend_type')['processing_latency_ms'].agg(['median', 'mean'])

    # Print results
    logger.info("Latency Statistics by Backend Type:")
    for backend, stats in latency_stats.iterrows():
        logger.info(f"  {backend}: Median={stats['median']:.2f}ms, Mean={stats['mean']:.2f}ms")

    # Create box plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='backend_type', y='processing_latency_ms', data=df, palette='viridis')

    # Add labels and title
    plt.xlabel('Backend Type')
    plt.ylabel('Processing Latency (ms)')
    plt.title('Processing Latency by Backend Type')

    # Save the plot
    output_file = os.path.join(output_dir, 'latency_boxplot.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    logger.info(f"Saved latency box plot to {output_file}")

    # Create violin plot for a different visualization
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='backend_type', y='processing_latency_ms', data=df, palette='viridis', inner='quartile')

    # Add labels and title
    plt.xlabel('Backend Type')
    plt.ylabel('Processing Latency (ms)')
    plt.title('Processing Latency Distribution by Backend Type')

    # Save the plot
    output_file = os.path.join(output_dir, 'latency_violinplot.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    logger.info(f"Saved latency violin plot to {output_file}")

    plt.close()

def plot_accuracy_over_time(df, output_dir):
    """
    Plot accuracy over time (simulated learning) by backend type.

    Args:
        df (pandas.DataFrame): DataFrame with simulation data
        output_dir (str): Directory to save the plot
    """
    # Filter for game_sort events if event_type column exists
    if 'event_type' in df.columns:
        game_sort_df = df[df['event_type'] == 'game_sort']
        if len(game_sort_df) > 0:
            logger.info(f"Filtering for game_sort events: {len(game_sort_df)} out of {len(df)} events")
            df = game_sort_df

    # Create time windows based on event_id
    if 'event_id' in df.columns:
        # Convert event_id to numeric if it's not already
        if not pd.api.types.is_numeric_dtype(df['event_id']):
            try:
                # Try to extract numeric part from event_id
                df['event_num'] = df['event_id'].str.extract(r'(\d+)').astype(float)
            except:
                # If that fails, create a sequence
                df['event_num'] = range(len(df))
        else:
            df['event_num'] = df['event_id']

        # Create time windows (bins of size 50)
        window_size = 50
        df['time_window'] = (df['event_num'] // window_size) * window_size
    elif 'timestamp' in df.columns:
        # If no event_id, use timestamp
        # Create time windows based on timestamp
        df['time_window'] = pd.to_datetime(df['timestamp']).dt.floor('1H')
    else:
        logger.warning("No event_id or timestamp column found, creating sequential windows")
        df['time_window'] = (df.index // 50) * 50

    # Group by backend_type and time_window, calculate mean is_correct
    accuracy_over_time = df.groupby(['backend_type', 'time_window'])['is_correct'].mean().reset_index()

    # Create line plot
    plt.figure(figsize=(14, 8))
    sns.lineplot(x='time_window', y='is_correct', hue='backend_type',
                 data=accuracy_over_time, markers=True, dashes=False, palette='viridis')

    # Add labels and title
    plt.xlabel('Time Window')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Time by Backend Type')
    plt.ylim(0, 1.0)  # Set y-axis from 0 to 1
    plt.legend(title='Backend Type')

    # Format x-axis based on the type of time_window
    if pd.api.types.is_datetime64_dtype(accuracy_over_time['time_window']):
        plt.gcf().autofmt_xdate()  # Rotate date labels

    # Save the plot
    output_file = os.path.join(output_dir, 'accuracy_over_time.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    logger.info(f"Saved accuracy over time plot to {output_file}")

    plt.close()

def plot_error_analysis(df, output_dir):
    """
    Analyze frequency of specific incorrect chosen_bin choices for critical item_ids.

    Args:
        df (pandas.DataFrame): DataFrame with simulation data
        output_dir (str): Directory to save the plot
    """
    # Check if we have the necessary columns
    if 'chosen_bin' not in df.columns or 'true_bin' not in df.columns or 'item_id' not in df.columns:
        logger.warning("Missing necessary columns for error analysis, skipping")
        return

    # Filter for incorrect choices
    error_df = df[df['is_correct'] == False].copy()

    if len(error_df) == 0:
        logger.warning("No errors found in the data, skipping error analysis")
        return

    # Get the top 10 items with the most errors
    top_error_items = error_df['item_id'].value_counts().head(10).index.tolist()

    # Filter for these items
    top_error_df = error_df[error_df['item_id'].isin(top_error_items)]

    # Create a heatmap of item_id vs chosen_bin for errors
    error_heatmap = pd.crosstab(
        top_error_df['item_id'],
        top_error_df['chosen_bin'],
        normalize='index'
    )

    plt.figure(figsize=(14, 10))
    sns.heatmap(error_heatmap, annot=True, cmap='YlOrRd', fmt='.1%', cbar=True)

    # Add labels and title
    plt.xlabel('Incorrectly Chosen Bin')
    plt.ylabel('Item ID')
    plt.title('Error Analysis: Frequency of Incorrect Bin Choices by Item')

    # Save the plot
    output_file = os.path.join(output_dir, 'error_analysis_heatmap.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    logger.info(f"Saved error analysis heatmap to {output_file}")

    # Also create a grouped bar chart of error counts by backend_type
    plt.figure(figsize=(14, 8))
    error_counts = pd.crosstab(error_df['item_id'], error_df['backend_type'])
    error_counts = error_counts.loc[error_counts.sum(axis=1).sort_values(ascending=False).head(10).index]

    error_counts.plot(kind='bar', figsize=(14, 8), colormap='viridis')

    # Add labels and title
    plt.xlabel('Item ID')
    plt.ylabel('Error Count')
    plt.title('Error Counts by Item ID and Backend Type')
    plt.legend(title='Backend Type')

    # Save the plot
    output_file = os.path.join(output_dir, 'error_counts_by_backend.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    logger.info(f"Saved error counts plot to {output_file}")

    plt.close()

def main():
    """Main function to analyze data and generate plots."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze simulation data and generate comparison plots')
    parser.add_argument('--data_file', required=True, help='Path to the combined data file (pickle, csv, or feather)')
    parser.add_argument('--output_dir', default='plots', help='Directory to save generated plot images')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    df = load_data(args.data_file)
    if df is None or len(df) == 0:
        logger.error("No data to analyze. Exiting.")
        return

    # Generate plots
    plot_overall_accuracy(df, args.output_dir)
    plot_latency(df, args.output_dir)
    plot_accuracy_over_time(df, args.output_dir)
    plot_error_analysis(df, args.output_dir)

    logger.info(f"Analysis complete. Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
