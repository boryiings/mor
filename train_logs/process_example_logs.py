#!/usr/bin/env python3
import re
import argparse
from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class TensorLogEntry:
    node_id: int
    tensor_name: str
    counter: int
    step: int
    tensor_dimension: str
    global_non_zero: int
    global_error: float
    relative_error: float
    histogram: List[int]

@dataclass
class AggregatedTensorStats:
    tensor_name: str
    counter: int  # Will store the common counter if all match, else -1
    step: int     # Will store the common step if all match, else -1
    tensor_dimension: str  # Will store the common dimension if all match, else "MISMATCH"
    entry_count: int
    mean_global_non_zero: float
    mean_global_error: float
    mean_relative_error: float
    aggregated_histogram: List[int]
    is_consistent: bool  # True if all counters, steps, and dimensions match

def parse_log_entry(line: str) -> TensorLogEntry:
    try:
        # Extract node ID from the beginning of the line
        node_id = int(line.split(':')[0].strip())
        
        # Extract other fields using regex
        tensor_match = re.search(r'Tensor\s+([\w\.]+)\s+at\s+counter\s+(\d+)\s+step\s+(\d+)', line)
        dimension_match = re.search(r'tensor dimension:\s+([\dx]+)', line)
        non_zero_match = re.search(r'global_non_zero:\s+(\d+)', line)
        global_error_match = re.search(r'global_error:\s+([\d\.]+)', line)
        relative_error_match = re.search(r'relative_error:\s+([\d\.]+)', line)
        histogram_match = re.search(r'histogram\s+\[([\d,\s]+)\]', line)
        
        if not all([tensor_match, non_zero_match, 
                   global_error_match, relative_error_match, histogram_match]):
            raise ValueError(f"Failed to match all required fields, tensor_match = {tensor_match}, dimension_match = {dimension_match}, non_zero_match = {non_zero_match}, global_error_match = {global_error_match}, relative_error_match = {relative_error_match}, histogram_match = {histogram_match}")

        # Parse histogram string to list of integers
        histogram_str = histogram_match.group(1)
        histogram = [int(x) for x in histogram_str.split(',')]

        return TensorLogEntry(
            node_id=node_id,
            tensor_name=tensor_match.group(1),
            counter=int(tensor_match.group(2)),
            step=int(tensor_match.group(3)),
            tensor_dimension=dimension_match.group(1) if dimension_match else None,
            global_non_zero=int(non_zero_match.group(1)),
            global_error=float(global_error_match.group(1)),
            relative_error=float(relative_error_match.group(1)),
            histogram=histogram
        )
    except Exception as e:
        print(f"Error parsing line: {str(e)}")
        return None

def process_log_file(log_file_path: str, target_counter: int = None) -> List[TensorLogEntry]:
    entries = []
    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                if 'Tensor' in line and 'at counter' in line:
                    if target_counter is not None:
                        counter_match = re.search(r'counter\s+(\d+)', line)
                        if counter_match:
                            current_counter = int(counter_match.group(1))
                            if current_counter < target_counter:
                                continue
                            if current_counter > target_counter:
                                break
                    
                    entry = parse_log_entry(line)
                    if entry:
                        entries.append(entry)
        return entries
    except FileNotFoundError:
        print(f"Error: File {log_file_path} not found")
        return []
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return []

def process_directory(root_dir: str, target_counter: int = None) -> Dict[str, List[TensorLogEntry]]:
    """Process log files in immediate subdirectories of the root directory."""
    all_entries = {}
    root_path = Path(root_dir)
    
    # Get all subdirectories and sort them chronologically
    subdirs = sorted([d for d in root_path.iterdir() if d.is_dir() and "heat" not in d.name], 
                    key=lambda x: x.name)
    
    # Iterate through sorted subdirectories
    for subdir in subdirs:
        # Look for log files in this subdirectory
        log_files = list(subdir.glob('*.log'))
        if log_files:
            print(f"\nProcessing subdirectory: {subdir.name}")
            for log_file in log_files:
                print(f"Processing file: {log_file.name}")
                entries = process_log_file(str(log_file), target_counter)
                if entries:
                    all_entries[str(log_file)] = entries
                    print(f"Found {len(entries)} entries in {log_file.name}")
                else:
                    print(f"No matching entries found in {log_file.name}")
    
    return all_entries

def aggregate_tensor_entries(entries: List[TensorLogEntry]) -> Dict[str, AggregatedTensorStats]:
    # Group entries by tensor name
    tensor_groups = defaultdict(list)
    for entry in entries:
        tensor_groups[entry.tensor_name].append(entry)
    
    aggregated_stats = {}
    
    for tensor_name, group_entries in tensor_groups.items():
        # Check consistency of counter, step, and dimension
        counters = {entry.counter for entry in group_entries}
        steps = {entry.step for entry in group_entries}
        dimensions = {entry.tensor_dimension for entry in group_entries}
        
        is_consistent = len(counters) == 1 and len(steps) == 1 and len(dimensions) == 1
        
        # Calculate means
        global_non_zeros = [entry.global_non_zero for entry in group_entries]
        global_errors = [entry.global_error for entry in group_entries]
        relative_errors = [entry.relative_error for entry in group_entries]
        
        # Aggregate histograms
        histogram_length = len(group_entries[0].histogram)
        aggregated_histogram = [0] * histogram_length
        for entry in group_entries:
            for i in range(histogram_length):
                aggregated_histogram[i] += entry.histogram[i]
        
        aggregated_stats[tensor_name] = AggregatedTensorStats(
            tensor_name=tensor_name,
            counter=list(counters)[0] if len(counters) == 1 else -1,
            step=list(steps)[0] if len(steps) == 1 else -1,
            tensor_dimension=list(dimensions)[0] if len(dimensions) == 1 else "MISMATCH",
            entry_count=len(group_entries),
            mean_global_non_zero=np.mean(global_non_zeros),
            mean_global_error=np.mean(global_errors),
            mean_relative_error=np.mean(relative_errors),
            aggregated_histogram=aggregated_histogram,
            is_consistent=is_consistent
        )
    
    return aggregated_stats

def create_histogram_heatmap(tensor_results: Dict[int, AggregatedTensorStats], 
                           output_dir: str):
    """Create heatmap for a tensor's histogram evolution."""
    # Get steps from the aggregated stats
    steps = sorted(tensor_results.keys())
    
    # Calculate figure height based on number of steps
    # Assume minimum height of 8, and add 0.3 inches per additional step beyond 20
    base_height = 8
    height = max(base_height, base_height + 0.3 * (len(steps) - 20))
    
    # Create a matrix where each row is a histogram for a step
    num_bins = 12  # Only use first 12 bins
    histogram_matrix = np.zeros((len(steps), num_bins))
    
    # Fill the matrix with histogram values
    for i, step in enumerate(steps):
        histogram_matrix[i] = tensor_results[step].aggregated_histogram[:num_bins]
    
    # Create normalized matrix
    normalized_matrix = histogram_matrix.copy()
    for i in range(len(steps)):
        row_sum = normalized_matrix[i].sum()
        if row_sum > 0:  # Avoid division by zero
            normalized_matrix[i] = normalized_matrix[i] / row_sum

    # Create x-axis labels (0.005, 0.01, ..., 0.06)
    x_labels = [f"{(i+1)*0.005:.3f}" for i in range(num_bins)]
    
    # Create the heatmap with dynamic height
    plt.figure(figsize=(12, height))
    sns.heatmap(normalized_matrix,
                xticklabels=x_labels,
                yticklabels=steps,
                cmap='YlOrRd',
                cbar_kws={'label': 'Normalized Count'},
                vmin=0,
                vmax=1)
    
    plt.title(f'Histogram Evolution for {next(iter(tensor_results.values())).tensor_name}')
    plt.xlabel('Threshold')
    plt.ylabel('Step')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()

def create_overall_distribution_heatmaps(tensor_results: Dict[str, Dict[int, AggregatedTensorStats]], 
                                       output_dir: str):
    """Create heatmaps showing overall histogram distribution for each tensor, both raw and normalized."""
    # Get all tensor names
    tensor_names = list(tensor_results.keys())
    num_bins = 12
    
    # Create a matrix where each row represents a tensor's summed histogram
    histogram_matrix = np.zeros((len(tensor_names), num_bins))
    
    # Fill the matrix with summed histogram values for each tensor
    for i, tensor_name in enumerate(tensor_names):
        # Sum histograms across all steps for this tensor
        summed_histogram = np.zeros(num_bins)
        for step_stats in tensor_results[tensor_name].values():
            summed_histogram += step_stats.aggregated_histogram[:num_bins]
        histogram_matrix[i] = summed_histogram
    
    # Create normalized matrix
    normalized_matrix = histogram_matrix.copy()
    for i in range(len(tensor_names)):
        row_sum = normalized_matrix[i].sum()
        if row_sum > 0:  # Avoid division by zero
            normalized_matrix[i] = normalized_matrix[i] / row_sum
    
    # Create x-axis labels
    x_labels = [f"{(i+1)*0.005:.3f}" for i in range(num_bins)]
    
    # Calculate figure height based on number of tensors
    base_height = 8
    height = max(base_height, base_height + 0.3 * (len(tensor_names) - 20))
    
    # Create the raw values heatmap
    plt.figure(figsize=(12, height))
    sns.heatmap(histogram_matrix, 
               xticklabels=x_labels,
               yticklabels=tensor_names,
               cmap='YlOrRd',
               cbar_kws={'label': 'Total Count'})
    
    plt.title('Overall Histogram Distribution by Tensor (Raw Values)')
    plt.xlabel('Threshold')
    plt.ylabel('Tensor Name')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_distribution_heatmap_raw.png'))
    plt.close()
    
    # Create the normalized heatmap
    plt.figure(figsize=(12, height))
    sns.heatmap(normalized_matrix, 
               xticklabels=x_labels,
               yticklabels=tensor_names,
               cmap='YlOrRd',
               cbar_kws={'label': 'Normalized Count'},
               vmin=0,
               vmax=1)
    
    plt.title('Overall Histogram Distribution by Tensor (Normalized)')
    plt.xlabel('Threshold')
    plt.ylabel('Tensor Name')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_distribution_heatmap_normalized.png'))
    plt.close()

def create_overall_histogram(tensor_results: Dict[str, Dict[int, AggregatedTensorStats]], 
                           output_dir: str,
                           histogram_only: bool = False):
    """Create a single histogram showing the sum of all histograms across all tensors and steps."""
    num_bins = 12
    overall_histogram = np.zeros(num_bins)
    
    # Sum histograms across all tensors and steps
    for tensor_name, step_results in tensor_results.items():
        for step, stats in step_results.items():
            overall_histogram += stats.aggregated_histogram[:num_bins]
    
    # Calculate normalized histogram (normalize by sum instead of max)
    total_sum = overall_histogram.sum()
    normalized_histogram = overall_histogram / total_sum if total_sum > 0 else overall_histogram
    
    # Create x-axis labels
    x_labels = [f"{(i+1)*0.005:.3f}" for i in range(num_bins)]
    
    # Print the values
    print("\nOverall Histogram Values:")
    print("Threshold  |  Raw Count  |  Normalized")
    print("-" * 40)
    for i, (x, v, n) in enumerate(zip(x_labels, overall_histogram, normalized_histogram)):
        print(f"{x:^9} | {int(v):^10} | {n:^10.3f}")
    
    if not histogram_only:
        # Create the bar plot
        plt.figure(figsize=(12, 6))
        plt.bar(x_labels, overall_histogram)
        
        plt.title('Overall Histogram Distribution (All Tensors Combined)')
        plt.xlabel('Threshold')
        plt.ylabel('Total Count')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add value labels on top of each bar (both raw and normalized)
        for i, (v, n) in enumerate(zip(overall_histogram, normalized_histogram)):
            plt.text(i, v, f'Raw: {int(v)}\nNorm: {n:.3f}', ha='center', va='bottom')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, 'overall_histogram_combined.png'))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract tensor information from log files in a directory.')
    parser.add_argument('directory', help='Path to the directory containing log files')
    parser.add_argument('--counter', type=int, default=500, help='Filter entries by counter value (default: 500)')
    parser.add_argument('--output', default='heatmaps', help='Output directory for heatmaps (default: heatmaps)')
    parser.add_argument('--histogram-only', action='store_true', 
                      help='Only print the overall histogram values, skip other plots')
    args = parser.parse_args()
    
    # Process all log files in the directory
    all_file_entries = process_directory(args.directory, args.counter)
    
    if all_file_entries:
        print(f"\nProcessed {len(all_file_entries)} log files")
        
        # Create a dictionary to store aggregated results for each tensor
        tensor_results = defaultdict(dict)  # tensor_name -> {step -> stats}
        
        # Process each file's entries and organize by tensor and step
        for file_path, log_entries in all_file_entries.items():
            aggregated_results = aggregate_tensor_entries(log_entries)
            
            for tensor_name, stats in aggregated_results.items():
                if stats.is_consistent and stats.step != -1:
                    tensor_results[tensor_name][stats.step] = stats
       
        if not args.histogram_only:
            # Create output directory
            os.makedirs(args.output, exist_ok=True)
            
            # Create individual heatmaps for each tensor
            print("\nGenerating individual tensor heatmaps...")
            for tensor_name, step_results in tensor_results.items():
                if step_results:
                    print(f"Creating heatmap for tensor: {tensor_name}")
                    create_histogram_heatmap(step_results, args.output)
                    safe_tensor_name = "".join(c if c.isalnum() else "_" for c in tensor_name)
                    plt.savefig(os.path.join(args.output, f'heatmap_{safe_tensor_name}.png'))
                    plt.close()
            
            # Create overall distribution heatmaps
            print("\nGenerating overall distribution heatmaps...")
            create_overall_distribution_heatmaps(tensor_results, args.output)
        
        # Generate overall histogram (values only or plot+values)
        print("\nGenerating overall histogram values...")
        create_overall_histogram(tensor_results, args.output, args.histogram_only)
        
        if not args.histogram_only:
            print(f"\nAll plots have been saved to the '{args.output}' directory")
    else:
        print(f"No log files found with matching entries in directory: {args.directory}")
