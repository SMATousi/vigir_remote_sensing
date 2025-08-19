#!/usr/bin/env python3
"""
Clustering vs Yield Map Evaluation Tool

This script evaluates clustering results against yield map masks by calculating
classification metrics including precision, recall, F1-score, and IoU for each cluster.

Author: Vigir Remote Sensing Team
Date: 2025-08-18
"""

import numpy as np
import rasterio
from rasterio.mask import mask
import argparse
import sys
import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns


class ClusteringYieldEvaluator:
    """
    Evaluates clustering results against yield map masks using various classification metrics.
    """
    
    def __init__(self, clustering_file: str, yield_file: str):
        """
        Initialize the evaluator with clustering and yield map files.
        
        Args:
            clustering_file: Path to the clustering results TIFF file
            yield_file: Path to the yield map TIFF file
        """
        self.clustering_file = clustering_file
        self.yield_file = yield_file
        self.clustering_data = None
        self.yield_data = None
        self.yield_mask = None
        self.transform = None
        self.crs = None
        
    def load_data(self) -> bool:
        """
        Load clustering and yield data from TIFF files.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load clustering results
            with rasterio.open(self.clustering_file) as src:
                self.clustering_data = src.read(1)
                self.transform = src.transform
                self.crs = src.crs
                print(f"Loaded clustering data: {self.clustering_data.shape}")
                print(f"Unique clusters: {np.unique(self.clustering_data[self.clustering_data > 0])}")
            
            # Load yield data
            with rasterio.open(self.yield_file) as src:
                self.yield_data = src.read(1)
                print(f"Loaded yield data: {self.yield_data.shape}")
                print(f"Yield range: {np.nanmin(self.yield_data):.2f} - {np.nanmax(self.yield_data):.2f}")
            
            # Check if dimensions match
            if self.clustering_data.shape != self.yield_data.shape:
                print("Warning: Clustering and yield data have different dimensions.")
                print(f"Clustering: {self.clustering_data.shape}, Yield: {self.yield_data.shape}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def create_yield_mask(self, yield_min: float, yield_max: float) -> np.ndarray:
        """
        Create a binary mask from yield data based on specified range.
        
        Args:
            yield_min: Minimum yield value for the mask
            yield_max: Maximum yield value for the mask
            
        Returns:
            np.ndarray: Binary mask (1 for pixels within range, 0 otherwise)
        """
        # Handle NaN values
        valid_yield = ~np.isnan(self.yield_data)
        
        # Create mask for yield range
        yield_range_mask = (self.yield_data >= yield_min) & (self.yield_data <= yield_max)
        
        # Combine with valid data mask
        self.yield_mask = (yield_range_mask & valid_yield).astype(int)
        
        mask_pixels = np.sum(self.yield_mask)
        total_valid_pixels = np.sum(valid_yield)
        
        print(f"Created yield mask for range [{yield_min:.2f}, {yield_max:.2f}]")
        print(f"Mask pixels: {mask_pixels} ({mask_pixels/total_valid_pixels*100:.1f}% of valid pixels)")
        
        return self.yield_mask
    
    def calculate_cluster_metrics(self, cluster_id: int) -> Dict[str, float]:
        """
        Calculate classification metrics for a specific cluster against the yield mask.
        
        Args:
            cluster_id: ID of the cluster to evaluate
            
        Returns:
            Dict containing precision, recall, F1-score, and IoU
        """
        # First, create a mask for valid yield data (non-NaN)
        valid_yield_mask = ~np.isnan(self.yield_data)
        
        if not np.any(valid_yield_mask):
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'iou': 0.0,
                'cluster_pixels_total': 0,
                'mask_pixels_total': 0,
                'intersection': 0,
                'union': 0
            }
        
        # Mask the clustering data to only include pixels with valid yield data
        masked_clustering = np.where(valid_yield_mask, self.clustering_data, 0)
        
        # Create binary mask for this specific cluster (only where yield data is valid)
        cluster_mask = (masked_clustering == cluster_id).astype(int)
        
        # Calculate basic counts
        cluster_pixels_total = np.sum(cluster_mask)
        mask_pixels_total = np.sum(self.yield_mask)
        intersection = np.sum(cluster_mask & self.yield_mask)  # Pixels that are both in cluster AND yield mask
        union = np.sum(cluster_mask | self.yield_mask)  # Pixels that are in cluster OR yield mask
        
        # Calculate metrics
        if cluster_pixels_total == 0:
            precision = 0.0
        else:
            precision = intersection / cluster_pixels_total
        
        if mask_pixels_total == 0:
            recall = 0.0
        else:
            recall = intersection / mask_pixels_total
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        if union == 0:
            iou = 0.0
        else:
            iou = intersection / union
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'iou': iou,
            'cluster_pixels_total': cluster_pixels_total,
            'mask_pixels_total': mask_pixels_total,
            'intersection': intersection,
            'union': union
        }
    
    def evaluate_all_clusters(self, yield_min: float, yield_max: float) -> pd.DataFrame:
        """
        Evaluate all clusters against the yield mask.
        
        Args:
            yield_min: Minimum yield value for the mask
            yield_max: Maximum yield value for the mask
            
        Returns:
            pd.DataFrame: Results for all clusters
        """
        # Create yield mask
        self.create_yield_mask(yield_min, yield_max)
        
        # Get unique cluster IDs (excluding 0 which is typically background)
        cluster_ids = np.unique(self.clustering_data)
        cluster_ids = cluster_ids[cluster_ids > 0]  # Remove background
        
        results = []
        
        print(f"\nEvaluating {len(cluster_ids)} clusters...")
        
        for cluster_id in cluster_ids:
            metrics = self.calculate_cluster_metrics(cluster_id)
            metrics['cluster_id'] = cluster_id
            results.append(metrics)
            
            print(f"Cluster {cluster_id}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}, "
                  f"IoU={metrics['iou']:.3f}")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        df = df[['cluster_id', 'precision', 'recall', 'f1_score', 'iou', 
                'cluster_pixels_total', 'mask_pixels_total', 'intersection', 'union']]
        
        return df
    
    def save_results(self, results_df: pd.DataFrame, output_file: str, 
                    yield_min: float, yield_max: float):
        """
        Save evaluation results to CSV file.
        
        Args:
            results_df: DataFrame containing evaluation results
            output_file: Path to output CSV file
            yield_min: Minimum yield value used for mask
            yield_max: Maximum yield value used for mask
        """
        # Add metadata as comments
        with open(output_file, 'w') as f:
            f.write(f"# Clustering vs Yield Map Evaluation Results\n")
            f.write(f"# Clustering file: {self.clustering_file}\n")
            f.write(f"# Yield file: {self.yield_file}\n")
            f.write(f"# Yield range: [{yield_min:.2f}, {yield_max:.2f}]\n")
            f.write(f"# Generated on: {pd.Timestamp.now()}\n")
            f.write("#\n")
        
        # Append DataFrame
        results_df.to_csv(output_file, mode='a', index=False)
        print(f"Results saved to: {output_file}")
    
    def create_visualization(self, results_df: pd.DataFrame, output_file: str):
        """
        Create visualization of evaluation metrics.
        
        Args:
            results_df: DataFrame containing evaluation results
            output_file: Path to output image file
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Precision plot
        axes[0, 0].bar(results_df['cluster_id'], results_df['precision'], 
                       color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Precision by Cluster')
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_ylim(0, 1)
        
        # Recall plot
        axes[0, 1].bar(results_df['cluster_id'], results_df['recall'], 
                       color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Recall by Cluster')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_ylim(0, 1)
        
        # F1-score plot
        axes[1, 0].bar(results_df['cluster_id'], results_df['f1_score'], 
                       color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('F1-Score by Cluster')
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_ylim(0, 1)
        
        # IoU plot
        axes[1, 1].bar(results_df['cluster_id'], results_df['iou'], 
                       color='gold', alpha=0.7)
        axes[1, 1].set_title('IoU by Cluster')
        axes[1, 1].set_xlabel('Cluster ID')
        axes[1, 1].set_ylabel('IoU')
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")
        plt.close()
    
    def create_overlay_visualization(self, output_file: str, yield_min: float, yield_max: float):
        """
        Create an overlay visualization showing clusters and yield mask together.
        
        Args:
            output_file: Path to output image file
            yield_min: Minimum yield value used for mask
            yield_max: Maximum yield value used for mask
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Mask invalid areas for better visualization
        valid_yield_mask = ~np.isnan(self.yield_data) 

        
        # 1. Original clustering results
        clustering_data = np.where(valid_yield_mask, self.clustering_data, np.nan)
        clustering_data = np.where(clustering_data>0, clustering_data, np.nan)
        im1 = axes[0, 0].imshow(clustering_data, cmap='tab10', interpolation='nearest')
        axes[0, 0].set_title('Clustering Results')
        axes[0, 0].set_xlabel('X (pixels)')
        axes[0, 0].set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=axes[0, 0], label='Cluster ID')
        
        # 2. Yield map with mask overlay
        yield_data = np.where(valid_yield_mask, self.yield_data, np.nan)
        im2 = axes[0, 1].imshow(yield_data, cmap='viridis', interpolation='nearest')
        axes[0, 1].set_title(f'Yield Map (Mask: {yield_min:.1f}-{yield_max:.1f})')
        axes[0, 1].set_xlabel('X (pixels)')
        axes[0, 1].set_ylabel('Y (pixels)')
        plt.colorbar(im2, ax=axes[0, 1], label='Yield')
        
        # Add mask contour
        mask_contour = np.where(self.yield_mask == 1, 1, 0)
        axes[0, 1].contour(mask_contour, levels=[0.5], colors='red', linewidths=2)
        
        # 3. Yield mask only
        mask_display = np.where(valid_yield_mask, self.yield_mask, np.nan)
        im3 = axes[1, 0].imshow(mask_display, cmap='Reds', interpolation='nearest', vmin=0, vmax=1)
        axes[1, 0].set_title(f'Yield Mask ({yield_min:.1f}-{yield_max:.1f})')
        axes[1, 0].set_xlabel('X (pixels)')
        axes[1, 0].set_ylabel('Y (pixels)')
        plt.colorbar(im3, ax=axes[1, 0], label='In Yield Range')
        
        # 4. Overlay: Clusters with yield mask boundaries
        # Create a composite visualization
        overlay_display = np.where(valid_yield_mask, self.clustering_data, np.nan)
        im4 = axes[1, 1].imshow(overlay_display, cmap='tab10', interpolation='nearest', alpha=0.7)
        
        # Add yield mask as contour overlay
        mask_contour = np.where(self.yield_mask == 1, 1, 0)
        axes[1, 1].contour(mask_contour, levels=[0.5], colors='red', linewidths=3, alpha=0.8)
        
        # Add yield mask as semi-transparent overlay
        mask_overlay = np.where(self.yield_mask == 1, 1, 0)
        axes[1, 1].imshow(mask_overlay, cmap='Reds', alpha=0.3, interpolation='nearest')
        
        axes[1, 1].set_title('Overlay: Clusters + Yield Mask')
        axes[1, 1].set_xlabel('X (pixels)')
        axes[1, 1].set_ylabel('Y (pixels)')
        plt.colorbar(im4, ax=axes[1, 1], label='Cluster ID')
        
        # Add legend for the overlay
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.3, label=f'Yield Range ({yield_min:.1f}-{yield_max:.1f})'),
                          Patch(facecolor='none', edgecolor='red', linewidth=3, label='Yield Mask Boundary')]
        axes[1, 1].legend(handles=legend_elements, loc='upper right')
        
        # Add overall title
        fig.suptitle('Clustering vs Yield Map Analysis', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Overlay visualization saved to: {output_file}")
        plt.close()
    
    def print_summary(self, results_df: pd.DataFrame):
        """
        Print summary statistics of the evaluation.
        
        Args:
            results_df: DataFrame containing evaluation results
        """
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        print(f"Total clusters evaluated: {len(results_df)}")
        print(f"Average precision: {results_df['precision'].mean():.3f} ± {results_df['precision'].std():.3f}")
        print(f"Average recall: {results_df['recall'].mean():.3f} ± {results_df['recall'].std():.3f}")
        print(f"Average F1-score: {results_df['f1_score'].mean():.3f} ± {results_df['f1_score'].std():.3f}")
        print(f"Average IoU: {results_df['iou'].mean():.3f} ± {results_df['iou'].std():.3f}")
        
        # Best performing clusters
        print(f"\nBest clusters by F1-score:")
        top_f1 = results_df.nlargest(3, 'f1_score')[['cluster_id', 'f1_score', 'precision', 'recall', 'iou']]
        print(top_f1.to_string(index=False))
        
        print(f"\nBest clusters by IoU:")
        top_iou = results_df.nlargest(3, 'iou')[['cluster_id', 'iou', 'precision', 'recall', 'f1_score']]
        print(top_iou.to_string(index=False))


def main():
    """Main function to run the clustering evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate clustering results against yield map masks')
    parser.add_argument('clustering_file', help='Path to clustering results TIFF file')
    parser.add_argument('yield_file', help='Path to yield map TIFF file')
    parser.add_argument('yield_min', type=float, help='Minimum yield value for mask')
    parser.add_argument('yield_max', type=float, help='Maximum yield value for mask')
    parser.add_argument('--output_dir', '-o', default='.', help='Output directory for results')
    parser.add_argument('--prefix', '-p', default='clustering_evaluation', 
                       help='Prefix for output files')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.clustering_file):
        print(f"Error: Clustering file not found: {args.clustering_file}")
        sys.exit(1)
    
    if not os.path.exists(args.yield_file):
        print(f"Error: Yield file not found: {args.yield_file}")
        sys.exit(1)
    
    # Validate yield range
    if args.yield_min >= args.yield_max:
        print(f"Error: yield_min ({args.yield_min}) must be less than yield_max ({args.yield_max})")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ClusteringYieldEvaluator(args.clustering_file, args.yield_file)
    
    # Load data
    if not evaluator.load_data():
        print("Failed to load data. Exiting.")
        sys.exit(1)
    
    # Evaluate all clusters
    results_df = evaluator.evaluate_all_clusters(args.yield_min, args.yield_max)
    
    # Generate output filenames
    csv_file = os.path.join(args.output_dir, f"{args.prefix}_results.csv")
    plot_file = os.path.join(args.output_dir, f"{args.prefix}_visualization.png")
    overlay_file = os.path.join(args.output_dir, f"{args.prefix}_overlay.png")
    
    # Save results
    evaluator.save_results(results_df, csv_file, args.yield_min, args.yield_max)
    evaluator.create_visualization(results_df, plot_file)
    evaluator.create_overlay_visualization(overlay_file, args.yield_min, args.yield_max)
    evaluator.print_summary(results_df)
    
    print(f"\nEvaluation complete! Check output files:")
    print(f"  - Results: {csv_file}")
    print(f"  - Metrics Visualization: {plot_file}")
    print(f"  - Overlay Visualization: {overlay_file}")


if __name__ == "__main__":
    main()
