import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings
import argparse
import os


class NDVICalculator:
    """
    A class for calculating and normalizing NDVI (Normalized Difference Vegetation Index)
    from multispectral TIFF files using rasterio.
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Attributes:
        red_band_idx (int): Index of the red band (default: 1)
        nir_band_idx (int): Index of the NIR band (default: 2)
    """
    
    def __init__(self, red_band_idx: int = 2, nir_band_idx: int = 3):
        """
        Initialize the NDVI Calculator.
        
        Args:
            red_band_idx (int): Band index for red channel (1-indexed)
            nir_band_idx (int): Band index for NIR channel (1-indexed)
        """
        self.red_band_idx = red_band_idx
        self.nir_band_idx = nir_band_idx
    
    def read_tiff(self, file_path: str) -> Tuple[rasterio.DatasetReader, dict]:
        """
        Read a TIFF file and return the dataset and metadata.
        
        Args:
            file_path (str): Path to the TIFF file
            
        Returns:
            Tuple[rasterio.DatasetReader, dict]: Dataset and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        dataset = rasterio.open(file_path)
        
        # Validate that we have enough bands
        if dataset.count < max(self.red_band_idx, self.nir_band_idx):
            raise ValueError(f"File has {dataset.count} bands, but need at least {max(self.red_band_idx, self.nir_band_idx)} bands")
        
        metadata = {
            'width': dataset.width,
            'height': dataset.height,
            'count': dataset.count,
            'dtype': dataset.dtypes[0],
            'crs': dataset.crs,
            'transform': dataset.transform,
            'bounds': dataset.bounds
        }
        
        return dataset, metadata
    
    def calculate_ndvi(self, dataset: rasterio.DatasetReader, 
                      window: Optional[Window] = None) -> np.ndarray:
        """
        Calculate NDVI for each pixel in the dataset.
        
        Args:
            dataset (rasterio.DatasetReader): Input raster dataset
            window (Optional[Window]): Specific window to read (default: entire image)
            
        Returns:
            np.ndarray: NDVI values for each pixel
        """
        # Read red and NIR bands
        red_band = dataset.read(self.red_band_idx, window=window).astype(np.float64)
        nir_band = dataset.read(self.nir_band_idx, window=window).astype(np.float64)
        
        # Handle division by zero and invalid values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            
            # Calculate NDVI: (NIR - Red) / (NIR + Red)
            numerator = nir_band - red_band
            denominator = nir_band + red_band
            
            # Avoid division by zero
            ndvi = np.divide(numerator, denominator, 
                           out=np.zeros_like(numerator), 
                           where=denominator != 0)
        
        # Set invalid pixels (where both bands are 0) to NaN
        ndvi[denominator == 0] = np.nan
        
        return ndvi
    
    def normalize_ndvi(self, ndvi: np.ndarray, 
                      method: str = 'min_max', 
                      percentile_range: Tuple[float, float] = (2, 98)) -> np.ndarray:
        """
        Normalize NDVI values across the field.
        
        Args:
            ndvi (np.ndarray): Raw NDVI values
            method (str): Normalization method ('min_max', 'percentile', 'z_score')
            percentile_range (Tuple[float, float]): Percentile range for percentile normalization
            
        Returns:
            np.ndarray: Normalized NDVI values
        """
        # Remove NaN values for calculations
        valid_mask = ~np.isnan(ndvi)
        valid_ndvi = ndvi[valid_mask]
        
        if len(valid_ndvi) == 0:
            return ndvi
        
        normalized_ndvi = ndvi.copy()
        
        if method == 'min_max':
            # Scale to [0, 1] range
            min_val = np.min(valid_ndvi)
            max_val = np.max(valid_ndvi)
            if max_val != min_val:
                normalized_ndvi[valid_mask] = (valid_ndvi - min_val) / (max_val - min_val)
            else:
                normalized_ndvi[valid_mask] = 0
                
        elif method == 'percentile':
            # Scale based on percentile range
            p_low, p_high = np.percentile(valid_ndvi, percentile_range)
            if p_high != p_low:
                normalized_ndvi[valid_mask] = np.clip(
                    (valid_ndvi - p_low) / (p_high - p_low), 0, 1
                )
            else:
                normalized_ndvi[valid_mask] = 0
                
        elif method == 'z_score':
            # Z-score normalization
            mean_val = np.mean(valid_ndvi)
            std_val = np.std(valid_ndvi)
            if std_val != 0:
                normalized_ndvi[valid_mask] = (valid_ndvi - mean_val) / std_val
            else:
                normalized_ndvi[valid_mask] = 0
                
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized_ndvi
    
    def process_tiff(self, input_path: str, 
                    output_path: Optional[str] = None,
                    normalization_method: str = 'min_max',
                    save_raw_ndvi: bool = False) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Complete processing pipeline: read TIFF, calculate NDVI, normalize values.
        
        Args:
            input_path (str): Path to input TIFF file
            output_path (Optional[str]): Path to save output TIFF (optional)
            normalization_method (str): Method for normalization
            save_raw_ndvi (bool): Whether to also save raw NDVI values
            
        Returns:
            Tuple[np.ndarray, np.ndarray, dict]: Raw NDVI, normalized NDVI, metadata
        """
        # Read the input file
        dataset, metadata = self.read_tiff(input_path)
        
        try:
            # Calculate NDVI
            print(f"Calculating NDVI using bands {self.red_band_idx} (Red) and {self.nir_band_idx} (NIR)...")
            raw_ndvi = self.calculate_ndvi(dataset)
            
            # Normalize NDVI
            print(f"Normalizing NDVI using {normalization_method} method...")
            normalized_ndvi = self.normalize_ndvi(raw_ndvi, method=normalization_method)
            
            # Print statistics
            self._print_statistics(raw_ndvi, normalized_ndvi)
            
            # Save outputs if requested
            if output_path:
                self._save_ndvi_outputs(raw_ndvi, normalized_ndvi, metadata, 
                                      output_path, save_raw_ndvi)
            
            return raw_ndvi, normalized_ndvi, metadata
            
        finally:
            dataset.close()
    
    def _print_statistics(self, raw_ndvi: np.ndarray, normalized_ndvi: np.ndarray):
        """Print statistics about the NDVI calculation."""
        valid_raw = raw_ndvi[~np.isnan(raw_ndvi)]
        valid_norm = normalized_ndvi[~np.isnan(normalized_ndvi)]
        
        print("\n=== NDVI Statistics ===")
        print(f"Valid pixels: {len(valid_raw):,} / {raw_ndvi.size:,} ({len(valid_raw)/raw_ndvi.size*100:.1f}%)")
        
        if len(valid_raw) > 0:
            print("\nRaw NDVI:")
            print(f"  Range: [{np.min(valid_raw):.4f}, {np.max(valid_raw):.4f}]")
            print(f"  Mean: {np.mean(valid_raw):.4f}")
            print(f"  Std: {np.std(valid_raw):.4f}")
            
            print("\nNormalized NDVI:")
            print(f"  Range: [{np.min(valid_norm):.4f}, {np.max(valid_norm):.4f}]")
            print(f"  Mean: {np.mean(valid_norm):.4f}")
            print(f"  Std: {np.std(valid_norm):.4f}")
    
    def _save_ndvi_outputs(self, raw_ndvi: np.ndarray, normalized_ndvi: np.ndarray,
                          metadata: dict, output_path: str, save_raw: bool):
        """Save NDVI outputs to TIFF files."""
        # Prepare output metadata
        out_meta = {
            'driver': 'GTiff',
            'height': metadata['height'],
            'width': metadata['width'],
            'count': 1,
            'dtype': 'float32',
            'crs': metadata['crs'],
            'transform': metadata['transform'],
            'compress': 'lzw',
            'nodata': np.nan
        }
        
        # Save normalized NDVI
        norm_path = output_path.replace('.tif', '_normalized.tif') if not output_path.endswith('_normalized.tif') else output_path
        with rasterio.open(norm_path, 'w', **out_meta) as dst:
            dst.write(normalized_ndvi.astype(np.float32), 1)
        print(f"Normalized NDVI saved to: {norm_path}")
        
        # Save raw NDVI if requested
        if save_raw:
            raw_path = output_path.replace('.tif', '_raw.tif').replace('_normalized', '_raw')
            with rasterio.open(raw_path, 'w', **out_meta) as dst:
                dst.write(raw_ndvi.astype(np.float32), 1)
            print(f"Raw NDVI saved to: {raw_path}")
    
    def visualize_ndvi(self, raw_ndvi: np.ndarray, normalized_ndvi: np.ndarray, 
                      save_plot: Optional[str] = None):
        """
        Create visualization plots for NDVI results.
        
        Args:
            raw_ndvi (np.ndarray): Raw NDVI values
            normalized_ndvi (np.ndarray): Normalized NDVI values
            save_plot (Optional[str]): Path to save the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Raw NDVI
        im1 = axes[0].imshow(raw_ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        axes[0].set_title('Raw NDVI')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Normalized NDVI
        im2 = axes[1].imshow(normalized_ndvi, cmap='RdYlGn')
        axes[1].set_title('Normalized NDVI')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Histogram comparison
        valid_raw = raw_ndvi[~np.isnan(raw_ndvi)]
        valid_norm = normalized_ndvi[~np.isnan(normalized_ndvi)]
        
        axes[2].hist(valid_raw, bins=50, alpha=0.5, label='Raw NDVI', density=True)
        axes[2].hist(valid_norm, bins=50, alpha=0.5, label='Normalized NDVI', density=True)
        axes[2].set_xlabel('NDVI Value')
        axes[2].set_ylabel('Density')
        axes[2].set_title('NDVI Distribution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(save_plot, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_plot}")
        
        plt.show()


def main():
    """Command-line interface for the NDVI calculator."""
    parser = argparse.ArgumentParser(description='Calculate and normalize NDVI from TIFF files')
    parser.add_argument('input', help='Input TIFF file path')
    parser.add_argument('-o', '--output', help='Output TIFF file path')
    parser.add_argument('--red-band', type=int, default=3, help='Red band index (1-indexed, default: 1)')
    parser.add_argument('--nir-band', type=int, default=4, help='NIR band index (1-indexed, default: 2)')
    parser.add_argument('--normalize', choices=['min_max', 'percentile', 'z_score'], 
                       default='min_max', help='Normalization method (default: min_max)')
    parser.add_argument('--save-raw', action='store_true', help='Also save raw NDVI values')
    parser.add_argument('--visualize', action='store_true', help='Show visualization plots')
    parser.add_argument('--save-plot', help='Save visualization plot to file')
    
    args = parser.parse_args()
    
    # Initialize calculator
    calculator = NDVICalculator(red_band_idx=args.red_band, nir_band_idx=args.nir_band)
    
    try:
        # Process the TIFF file
        raw_ndvi, normalized_ndvi, metadata = calculator.process_tiff(
            input_path=args.input,
            output_path=args.output,
            normalization_method=args.normalize,
            save_raw_ndvi=args.save_raw
        )
        
        # Visualize if requested
        if args.visualize or args.save_plot:
            calculator.visualize_ndvi(raw_ndvi, normalized_ndvi, args.save_plot)
        
        print("\nNDVI calculation completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())