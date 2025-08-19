#!/usr/bin/env python3
"""
Example script showing how to use the clustering evaluation tool.

This script demonstrates how to evaluate clustering results against yield maps
using different yield ranges.
"""

import os
import sys
from clustering_yield_evaluation import ClusteringYieldEvaluator

def main():
    """Example usage of the clustering evaluation tool."""
    
    # Example file paths (update these to your actual files)
    clustering_file = "clustered_results.tif"
    yield_file = "your_yield_map.tif"  # Replace with your yield map file
    
    # Check if files exist
    if not os.path.exists(clustering_file):
        print(f"Clustering file not found: {clustering_file}")
        print("Available clustering files in current directory:")
        for f in os.listdir('.'):
            if f.endswith('.tif') and 'cluster' in f.lower():
                print(f"  - {f}")
        return
    
    if not os.path.exists(yield_file):
        print(f"Yield file not found: {yield_file}")
        print("Please update the 'yield_file' variable with your actual yield map file path.")
        return
    
    # Example yield ranges to evaluate
    yield_ranges = [
        (150, 200),  # High yield range
        (100, 150),  # Medium yield range
        (50, 100),   # Low yield range
    ]
    
    print("Running clustering evaluation examples...")
    
    for i, (yield_min, yield_max) in enumerate(yield_ranges):
        print(f"\n{'='*60}")
        print(f"EVALUATION {i+1}: Yield range [{yield_min}, {yield_max}]")
        print('='*60)
        
        # Initialize evaluator
        evaluator = ClusteringYieldEvaluator(clustering_file, yield_file)
        
        # Load data
        if not evaluator.load_data():
            print(f"Failed to load data for evaluation {i+1}")
            continue
        
        # Evaluate clusters
        results_df = evaluator.evaluate_all_clusters(yield_min, yield_max)
        
        # Save results with unique names
        output_prefix = f"evaluation_yield_{yield_min}_{yield_max}"
        csv_file = f"{output_prefix}_results.csv"
        plot_file = f"{output_prefix}_visualization.png"
        overlay_file = f"{output_prefix}_overlay.png"
        
        evaluator.save_results(results_df, csv_file, yield_min, yield_max)
        evaluator.create_visualization(results_df, plot_file)
        evaluator.create_overlay_visualization(overlay_file, yield_min, yield_max)
        evaluator.print_summary(results_df)
        
        print(f"Results saved:")
        print(f"  - CSV: {csv_file}")
        print(f"  - Metrics Plot: {plot_file}")
        print(f"  - Overlay Plot: {overlay_file}")

if __name__ == "__main__":
    main()
