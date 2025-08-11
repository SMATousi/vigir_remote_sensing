#!/usr/bin/env python3
"""
Spatial Clustering of Stacked TIFF Files

This script stacks multiple TIFF files and performs spatial clustering using DBSCAN algorithm.
DBSCAN automatically determines the number of clusters based on data density.

Author: Generated for VIGIR Remote Sensing Project
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as ResamplingEnum
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)


class SpatialTIFFClusterer:
    """
    A class for performing spatial clustering on stacked TIFF files.
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5, standardize: bool = True, 
                 normalize_layers: bool = True):
        """
        Initialize the clusterer.
        
        Args:
            eps: The maximum distance between two samples for one to be considered
                 as in the neighborhood of the other (DBSCAN parameter)
            min_samples: The number of samples in a neighborhood for a point to be
                        considered as a core point (DBSCAN parameter)
            standardize: Whether to standardize the data before clustering
            normalize_layers: Whether to normalize each layer individually before stacking
        """
        self.eps = eps
        self.min_samples = min_samples
        self.standardize = standardize
        self.normalize_layers = normalize_layers
        self.scaler = StandardScaler() if standardize else None
        self.clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        
    def get_lowest_resolution_metadata(self, tiff_paths: List[str]) -> Tuple[dict, int]:
        """
        Determine the lowest resolution (largest pixel size) among all TIFF files.
        
        Args:
            tiff_paths: List of paths to TIFF files
            
        Returns:
            Tuple of (reference_metadata, reference_index) where reference_metadata
            contains the metadata of the lowest resolution file
        """
        print("Analyzing resolutions of all TIFF files...")
        
        lowest_resolution = 0
        reference_idx = 0
        reference_metadata = None
        
        for i, tiff_path in enumerate(tiff_paths):
            if not os.path.exists(tiff_path):
                raise FileNotFoundError(f"TIFF file not found: {tiff_path}")
                
            with rasterio.open(tiff_path) as src:
                # Calculate pixel size (resolution)
                pixel_size_x = abs(src.transform[0])
                pixel_size_y = abs(src.transform[4])
                avg_pixel_size = (pixel_size_x + pixel_size_y) / 2
                
                print(f"  {os.path.basename(tiff_path)}: {src.height}x{src.width}, "
                      f"pixel size: {avg_pixel_size:.6f}")
                
                # Keep track of the lowest resolution (largest pixel size)
                if avg_pixel_size > lowest_resolution:
                    lowest_resolution = avg_pixel_size
                    reference_idx = i
                    reference_metadata = {
                        'height': src.height,
                        'width': src.width,
                        'transform': src.transform,
                        'crs': src.crs,
                        'dtype': src.dtypes[0],
                        'bounds': src.bounds
                    }
        
        print(f"\nUsing {os.path.basename(tiff_paths[reference_idx])} as reference "
              f"(lowest resolution: {lowest_resolution:.6f})")
        
        return reference_metadata, reference_idx
    
    def resample_to_reference(self, src_path: str, reference_metadata: dict) -> np.ndarray:
        """
        Resample a TIFF file to match the reference resolution and extent.
        
        Args:
            src_path: Path to the source TIFF file
            reference_metadata: Metadata of the reference (lowest resolution) file
            
        Returns:
            Resampled data array
        """
        with rasterio.open(src_path) as src:
            # Create destination array
            dst_array = np.zeros(
                (reference_metadata['height'], reference_metadata['width']),
                dtype=np.float32
            )
            
            # Reproject/resample the source data to match reference
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=reference_metadata['transform'],
                dst_crs=reference_metadata['crs'],
                resampling=Resampling.bilinear
            )
            
            # Handle nodata values
            if src.nodata is not None:
                dst_array[dst_array == src.nodata] = np.nan
                
            return dst_array
    
    def normalize_layer(self, data: np.ndarray, method: str = 'min_max') -> np.ndarray:
        """
        Normalize a single data layer.
        
        Args:
            data: 2D numpy array to normalize
            method: Normalization method ('min_max', 'z_score', or 'robust')
            
        Returns:
            Normalized data array
        """
        # Create a copy to avoid modifying original data
        normalized_data = data.copy()
        
        # Get valid (non-NaN) values for normalization
        valid_mask = ~np.isnan(data)
        valid_values = data[valid_mask]
        
        if len(valid_values) == 0:
            print("    Warning: No valid values found for normalization")
            return normalized_data
        
        if method == 'min_max':
            # Min-max normalization: scale to [0, 1]
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)
            if max_val > min_val:
                normalized_data[valid_mask] = (data[valid_mask] - min_val) / (max_val - min_val)
            else:
                normalized_data[valid_mask] = 0.5  # All values are the same
                
        elif method == 'z_score':
            # Z-score normalization: mean=0, std=1
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            if std_val > 0:
                normalized_data[valid_mask] = (data[valid_mask] - mean_val) / std_val
            else:
                normalized_data[valid_mask] = 0  # All values are the same
                
        elif method == 'robust':
            # Robust normalization using percentiles
            p25 = np.percentile(valid_values, 25)
            p75 = np.percentile(valid_values, 75)
            median_val = np.median(valid_values)
            iqr = p75 - p25
            if iqr > 0:
                normalized_data[valid_mask] = (data[valid_mask] - median_val) / iqr
            else:
                normalized_data[valid_mask] = 0  # All values are the same
        
        return normalized_data
    
    def load_and_stack_tiffs(self, tiff_paths: List[str]) -> Tuple[np.ndarray, dict]:
        """
        Load and stack multiple TIFF files, resizing them to the lowest resolution.
        
        Args:
            tiff_paths: List of paths to TIFF files
            
        Returns:
            Tuple of (stacked_data, metadata) where stacked_data is 3D array
            with shape (height, width, n_bands) and metadata contains rasterio info
        """
        print(f"Loading and stacking {len(tiff_paths)} TIFF files...")
        
        # Get the reference metadata (lowest resolution)
        reference_metadata, reference_idx = self.get_lowest_resolution_metadata(tiff_paths)
        
        # Initialize the stacked array
        height = reference_metadata['height']
        width = reference_metadata['width']
        stacked_data = np.zeros((height, width, len(tiff_paths)), dtype=np.float32)
        
        # Load and resample each TIFF file
        for i, tiff_path in enumerate(tiff_paths):
            print(f"  Processing: {os.path.basename(tiff_path)}")
            
            if i == reference_idx:
                # This is the reference file, load directly
                with rasterio.open(tiff_path) as src:
                    band_data = src.read(1).astype(np.float32)
                    if src.nodata is not None:
                        band_data[band_data == src.nodata] = np.nan
                    
                    # Normalize if requested
                    if self.normalize_layers:
                        original_stats = f"range: [{np.nanmin(band_data):.3f}, {np.nanmax(band_data):.3f}]"
                        band_data = self.normalize_layer(band_data)
                        normalized_stats = f"range: [{np.nanmin(band_data):.3f}, {np.nanmax(band_data):.3f}]"
                        print(f"    Loaded directly (reference resolution), normalized: {original_stats} -> {normalized_stats}")
                    else:
                        print(f"    Loaded directly (reference resolution)")
                    
                    stacked_data[:, :, i] = band_data
            else:
                # Resample to match reference
                resampled_data = self.resample_to_reference(tiff_path, reference_metadata)
                
                # Normalize if requested
                if self.normalize_layers:
                    original_stats = f"range: [{np.nanmin(resampled_data):.3f}, {np.nanmax(resampled_data):.3f}]"
                    resampled_data = self.normalize_layer(resampled_data)
                    normalized_stats = f"range: [{np.nanmin(resampled_data):.3f}, {np.nanmax(resampled_data):.3f}]"
                    print(f"    Resampled and normalized: {original_stats} -> {normalized_stats}")
                else:
                    print(f"    Resampled to match reference resolution")
                
                stacked_data[:, :, i] = resampled_data
        
        metadata = {
            'height': height,
            'width': width,
            'transform': reference_metadata['transform'],
            'crs': reference_metadata['crs'],
            'dtype': reference_metadata['dtype'],
            'n_bands': len(tiff_paths)
        }
        
        print(f"\nFinal stacked data shape: {stacked_data.shape}")
        return stacked_data, metadata
    
    def prepare_data_for_clustering(self, stacked_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare stacked data for clustering by reshaping and handling NaN values.
        
        Args:
            stacked_data: 3D array with shape (height, width, n_bands)
            
        Returns:
            Tuple of (reshaped_data, valid_mask) where reshaped_data is 2D array
            suitable for clustering and valid_mask indicates valid pixels
        """
        height, width, n_bands = stacked_data.shape
        
        # Reshape to 2D: (n_pixels, n_bands)
        reshaped_data = stacked_data.reshape(-1, n_bands)
        
        # Create mask for valid pixels (no NaN values)
        valid_mask = ~np.isnan(reshaped_data).any(axis=1)
        
        # Extract valid pixels only
        valid_data = reshaped_data[valid_mask]
        
        print(f"Data shape: {stacked_data.shape}")
        print(f"Valid pixels: {valid_mask.sum():,} / {len(valid_mask):,} "
              f"({100 * valid_mask.sum() / len(valid_mask):.1f}%)")
        
        return valid_data, valid_mask
    
    def perform_clustering(self, data: np.ndarray) -> np.ndarray:
        """
        Perform DBSCAN clustering on the data.
        
        Args:
            data: 2D array with shape (n_samples, n_features)
            
        Returns:
            1D array of cluster labels
        """
        print("Performing clustering...")
        
        # Standardize data if requested
        if self.standardize:
            print("  Standardizing data...")
            data_scaled = self.scaler.fit_transform(data)
        else:
            data_scaled = data
        
        # Perform clustering
        print(f"  Running DBSCAN (eps={self.eps}, min_samples={self.min_samples})...")
        cluster_labels = self.clusterer.fit_predict(data_scaled)
        
        # Print clustering results
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"  Found {n_clusters} clusters")
        print(f"  Noise points: {n_noise:,} ({100 * n_noise / len(cluster_labels):.1f}%)")
        
        return cluster_labels
    
    def save_cluster_results(self, cluster_labels: np.ndarray, valid_mask: np.ndarray, 
                           metadata: dict, output_path: str):
        """
        Save clustering results as a TIFF file.
        
        Args:
            cluster_labels: 1D array of cluster labels for valid pixels
            valid_mask: Boolean mask indicating valid pixels
            metadata: Metadata from original TIFF files
            output_path: Path to save the output TIFF file
        """
        print(f"Saving results to: {output_path}")
        
        # Create full cluster array with nodata for invalid pixels
        full_cluster_array = np.full(len(valid_mask), -9999, dtype=np.int32)
        full_cluster_array[valid_mask] = cluster_labels
        
        # Reshape back to 2D
        cluster_image = full_cluster_array.reshape(metadata['height'], metadata['width'])
        
        # Save as TIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=metadata['height'],
            width=metadata['width'],
            count=1,
            dtype=np.int32,
            crs=metadata['crs'],
            transform=metadata['transform'],
            nodata=-9999,
            compress='lzw'
        ) as dst:
            dst.write(cluster_image, 1)
        
        print(f"Clustering results saved successfully!")
    
    def create_visualization(self, cluster_labels: np.ndarray, valid_mask: np.ndarray,
                           metadata: dict, output_path: str):
        """
        Create a visualization of the clustering results.
        
        Args:
            cluster_labels: 1D array of cluster labels for valid pixels
            valid_mask: Boolean mask indicating valid pixels
            metadata: Metadata from original TIFF files
            output_path: Path to save the visualization
        """
        # Create full cluster array
        full_cluster_array = np.full(len(valid_mask), np.nan)
        full_cluster_array[valid_mask] = cluster_labels
        
        # Reshape back to 2D
        cluster_image = full_cluster_array.reshape(metadata['height'], metadata['width'])
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        plt.imshow(cluster_image, cmap='tab20', interpolation='nearest')
        plt.colorbar(label='Cluster ID')
        plt.title('Spatial Clustering Results')
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        
        # Add statistics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        plt.figtext(0.02, 0.02, f'Clusters: {n_clusters}, Noise: {n_noise:,}', 
                   fontsize=10, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {output_path}")


def main():
    """
    Main function to run the spatial clustering workflow.
    """
    # =============================================================================
    # CONFIGURATION - MODIFY THESE PATHS TO YOUR TIFF FILES
    # =============================================================================
    
    # List of TIFF files to stack and cluster
    # Replace these paths with your actual TIFF file paths
    input_tiff_files = [
        "Ellis-Field/clipped_slope.tif",
        "Ellis-Field/rasterized_soil_survey.tif",
        "ndvi_temporal_statistics/ndvi_management_zones_2016_2024.tif",
        "ndvi_temporal_statistics/ndvi_median_yield_ranking_2016_2024.tif",
        "ndvi_temporal_statistics/ndvi_standard_deviation_2016_2024.tif",
    ]
    
    # Output paths
    output_cluster_tiff = "clustered_results.tif"
    output_visualization = "clustering_visualization.png"
    
    # Clustering parameters
    eps = 0.5          # Maximum distance between samples in a cluster
    min_samples = 100    # Minimum samples required to form a cluster
    standardize = True  # Whether to standardize the data
    normalize_layers = True  # Whether to normalize each layer before stacking
    
    # =============================================================================
    # END CONFIGURATION
    # =============================================================================
    
    try:
        # Initialize clusterer
        clusterer = SpatialTIFFClusterer(
            eps=eps, 
            min_samples=min_samples, 
            standardize=standardize,
            normalize_layers=normalize_layers
        )
        
        # Load and stack TIFF files
        stacked_data, metadata = clusterer.load_and_stack_tiffs(input_tiff_files)
        
        # Prepare data for clustering
        valid_data, valid_mask = clusterer.prepare_data_for_clustering(stacked_data)
        
        if len(valid_data) == 0:
            raise ValueError("No valid data found for clustering!")
        
        # Perform clustering
        cluster_labels = clusterer.perform_clustering(valid_data)
        
        # Save results
        clusterer.save_cluster_results(cluster_labels, valid_mask, metadata, output_cluster_tiff)
        
        # Create visualization
        clusterer.create_visualization(cluster_labels, valid_mask, metadata, output_visualization)
        
        print("\n" + "="*60)
        print("CLUSTERING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Input files: {len(input_tiff_files)}")
        print(f"Output cluster TIFF: {output_cluster_tiff}")
        print(f"Output visualization: {output_visualization}")
        
    except Exception as e:
        print(f"Error during clustering: {str(e)}")
        raise


if __name__ == "__main__":
    main()
