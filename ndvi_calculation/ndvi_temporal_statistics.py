#!/usr/bin/env python3
"""
NDVI Temporal Statistics Calculator

This script processes NDVI TIFF files across multiple years to calculate:
1. Median NDVI values across years (yield ranking)
2. Variability ranking (95th percentile - 5th percentile)
3. Standard deviation of NDVI values across years

Author: Generated for VIGIR Remote Sensing Project
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import glob
from pathlib import Path
import logging
from typing import List, Tuple, Dict
import argparse
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NDVITemporalAnalyzer:
    """
    Analyzes NDVI data across multiple years to calculate temporal statistics.
    """
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the temporal analyzer.
        
        Args:
            input_dir (str): Directory containing NDVI TIFF files
            output_dir (str): Directory to save output statistics files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Find all NDVI TIFF files
        self.ndvi_files = self._find_ndvi_files()
        logger.info(f"Found {len(self.ndvi_files)} NDVI files")
        
    def _find_ndvi_files(self) -> List[Path]:
        """Find all NDVI TIFF files in the input directory."""
        patterns = ['*ndvi*.tif', '*NDVI*.tif', '*ndvi*.tiff', '*NDVI*.tiff']
        files = []
        
        for pattern in patterns:
            files.extend(self.input_dir.glob(pattern))
        
        # Filter out auxiliary files
        files = [f for f in files if not f.name.endswith('.aux.xml')]
        
        # Sort by filename (which should include year)
        files.sort()
        
        return files
    
    def _extract_year_from_filename(self, filepath: Path) -> str:
        """Extract year from filename."""
        filename = filepath.stem
        # Look for 4-digit year in filename
        import re
        year_match = re.search(r'(20\d{2})', filename)
        if year_match:
            return year_match.group(1)
        else:
            return filename.split('_')[0]  # Fallback to first part
    
    def _load_ndvi_data(self) -> Tuple[np.ndarray, Dict, List[str]]:
        """
        Load all NDVI files and stack them into a 3D array.
        
        Returns:
            Tuple containing:
            - 3D numpy array (height, width, years)
            - Metadata from first file
            - List of years
        """
        logger.info("Loading NDVI data...")
        
        # Read first file to get dimensions and metadata
        with rasterio.open(self.ndvi_files[0]) as src:
            first_data = src.read(1)
            profile = src.profile
            height, width = first_data.shape
        
        # Initialize 3D array
        num_years = len(self.ndvi_files)
        ndvi_stack = np.full((height, width, num_years), np.nan, dtype=np.float32)
        years = []
        
        # Load all files
        for i, filepath in enumerate(self.ndvi_files):
            year = self._extract_year_from_filename(filepath)
            years.append(year)
            
            try:
                with rasterio.open(filepath) as src:
                    data = src.read(1).astype(np.float32)
                    
                    # Data is already normalized - no scaling needed
                    # Just handle nodata values and extreme outliers
                    nodata = src.nodata
                    if nodata is not None:
                        data = np.where(data == nodata, np.nan, data)
                    
                    # Mask extreme outliers (beyond reasonable NDVI range)
                    data = np.where((data < -2) | (data > 2), np.nan, data)
                    
                    ndvi_stack[:, :, i] = data
                    logger.info(f"Loaded {year}: {filepath.name} (range: {np.nanmin(data):.3f} to {np.nanmax(data):.3f})")
                    
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
                continue
        
        logger.info(f"Successfully loaded {num_years} years of data")
        return ndvi_stack, profile, years
    
    def calculate_median_yield_ranking(self, ndvi_stack: np.ndarray) -> np.ndarray:
        """
        Calculate median NDVI values across years for yield ranking.
        
        Args:
            ndvi_stack (np.ndarray): 3D array of NDVI data (height, width, years)
            
        Returns:
            np.ndarray: 2D array of median values
        """
        logger.info("Calculating median yield ranking...")
        
        # Calculate median across years (axis=2)
        median_values = np.nanmedian(ndvi_stack, axis=2)
        
        # Handle pixels with all NaN values
        median_values = np.where(np.isnan(median_values), -999, median_values)
        
        return median_values.astype(np.float32)
    
    def calculate_variability_ranking(self, ndvi_stack: np.ndarray) -> np.ndarray:
        """
        Calculate variability ranking based on (95th percentile - 5th percentile).
        
        Args:
            ndvi_stack (np.ndarray): 3D array of NDVI data (height, width, years)
            
        Returns:
            np.ndarray: 2D array of variability values
        """
        logger.info("Calculating variability ranking (95th - 5th percentile)...")
        
        height, width, years = ndvi_stack.shape
        variability = np.full((height, width), np.nan, dtype=np.float32)
        
        for i in range(height):
            for j in range(width):
                pixel_values = ndvi_stack[i, j, :]
                valid_values = pixel_values[~np.isnan(pixel_values)]
                
                if len(valid_values) >= 3:  # Need at least 3 values for meaningful percentiles
                    p95 = np.percentile(valid_values, 95)
                    p5 = np.percentile(valid_values, 5)
                    variability[i, j] = p95 - p5
        
        # Handle pixels with insufficient data
        variability = np.where(np.isnan(variability), -999, variability)
        
        return variability
    
    def calculate_standard_deviation(self, ndvi_stack: np.ndarray) -> np.ndarray:
        """
        Calculate standard deviation of NDVI values across years.
        
        Args:
            ndvi_stack (np.ndarray): 3D array of NDVI data (height, width, years)
            
        Returns:
            np.ndarray: 2D array of standard deviation values
        """
        logger.info("Calculating standard deviation across years...")
        
        # Calculate standard deviation across years (axis=2)
        std_values = np.nanstd(ndvi_stack, axis=2, ddof=1)
        
        # Handle pixels with insufficient data
        std_values = np.where(np.isnan(std_values), -999, std_values)
        
        return std_values.astype(np.float32)
    
    def calculate_absolute_normalization(self, ndvi_stack: np.ndarray) -> np.ndarray:
        """
        Calculate absolute normalization values across all years.
        This normalizes each pixel's performance relative to the entire field across all years.
        
        Args:
            ndvi_stack (np.ndarray): 3D array of NDVI data (height, width, years)
            
        Returns:
            np.ndarray: 2D array of absolute normalized values
        """
        logger.info("Calculating absolute normalization across all years...")
        
        height, width, years = ndvi_stack.shape
        
        # Flatten spatial dimensions to get all pixel values across all years
        all_values = ndvi_stack.reshape(-1, years)
        
        # Remove rows with all NaN values
        valid_mask = ~np.all(np.isnan(all_values), axis=1)
        valid_values = all_values[valid_mask]
        
        if len(valid_values) == 0:
            logger.warning("No valid values found for normalization")
            return np.full((height, width), -999, dtype=np.float32)
        
        # Calculate mean NDVI for each pixel across years
        pixel_means = np.nanmean(valid_values, axis=1)
        
        # Calculate overall statistics across all pixels and years
        overall_mean = np.nanmean(pixel_means)
        overall_std = np.nanstd(pixel_means)
        
        logger.info(f"Overall field statistics - Mean: {overall_mean:.4f}, Std: {overall_std:.4f}")
        
        # Normalize each pixel's mean relative to the overall field
        normalized_means = (pixel_means - overall_mean) / overall_std if overall_std > 0 else np.zeros_like(pixel_means)
        
        # Reconstruct 2D array
        result = np.full((height, width), -999, dtype=np.float32)
        result_flat = result.reshape(-1)
        result_flat[valid_mask] = normalized_means
        
        return result.reshape(height, width)
    
    def calculate_temporal_trend(self, ndvi_stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate temporal trend (slope) for each pixel across years.
        
        Args:
            ndvi_stack (np.ndarray): 3D array of NDVI data (height, width, years)
            
        Returns:
            Tuple of (slope_array, r_squared_array)
        """
        logger.info("Calculating temporal trends...")
        
        height, width, num_years = ndvi_stack.shape
        slopes = np.full((height, width), -999, dtype=np.float32)
        r_squared = np.full((height, width), -999, dtype=np.float32)
        
        # Create year array for regression
        years_array = np.arange(num_years)
        
        for i in range(height):
            for j in range(width):
                pixel_values = ndvi_stack[i, j, :]
                valid_mask = ~np.isnan(pixel_values)
                
                if np.sum(valid_mask) >= 3:  # Need at least 3 points for meaningful trend
                    valid_years = years_array[valid_mask]
                    valid_ndvi = pixel_values[valid_mask]
                    
                    # Linear regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(valid_years, valid_ndvi)
                    slopes[i, j] = slope
                    r_squared[i, j] = r_value ** 2
        
        return slopes, r_squared
    
    def classify_management_zones(self, median_yield: np.ndarray, std_deviation: np.ndarray, 
                                absolute_norm: np.ndarray, trend_slope: np.ndarray,
                                trend_r2: np.ndarray) -> np.ndarray:
        """
        Classify pixels into management zones based on NDVI characteristics.
        
        Zone Classifications:
        1 = Stable High (high median performance)
        2 = Stable Low (low median performance) 
        3 = Variable (high standard deviation)
        4 = Declining Trend (significant negative slope)
        5 = Increasing Trend (significant positive slope)
        0 = Invalid/NoData (pixels with insufficient data)
        
        Args:
            median_yield: Median NDVI values
            std_deviation: Standard deviation values
            absolute_norm: Absolute normalization values
            trend_slope: Temporal trend slopes
            trend_r2: R-squared values for trends
            
        Returns:
            np.ndarray: 2D array of zone classifications
        """
        logger.info("Classifying management zones...")
        
        height, width = median_yield.shape
        zones = np.zeros((height, width), dtype=np.uint8)
        
        # Calculate thresholds based on data distribution
        # Only use valid data (not -999) for threshold calculation
        valid_median = median_yield[median_yield != -999]
        valid_std = std_deviation[std_deviation != -999]
        valid_abs_norm = absolute_norm[absolute_norm != -999]
        valid_slope = trend_slope[trend_slope != -999]
        valid_r2 = trend_r2[trend_r2 != -999]
        
        if len(valid_median) == 0:
            logger.warning("No valid data for zone classification")
            return zones
        
        # Define thresholds
        median_high_thresh = np.percentile(valid_median, 75)
        median_low_thresh = np.percentile(valid_median, 25)
        std_high_thresh = np.percentile(valid_std, 75)
        abs_norm_high_thresh = np.percentile(valid_abs_norm, 75) if len(valid_abs_norm) > 0 else 0
        abs_norm_low_thresh = np.percentile(valid_abs_norm, 25) if len(valid_abs_norm) > 0 else 0
        
        # Trend significance thresholds
        slope_decline_thresh = np.percentile(valid_slope, 10) if len(valid_slope) > 0 else -0.01
        slope_increase_thresh = np.percentile(valid_slope, 90) if len(valid_slope) > 0 else 0.01
        r2_significance = 0.5  # R² threshold for significant trends
        
        logger.info(f"Classification thresholds:")
        logger.info(f"  Median High: {median_high_thresh:.3f}, Low: {median_low_thresh:.3f}")
        logger.info(f"  Std High: {std_high_thresh:.3f}")
        logger.info(f"  Abs Norm High: {abs_norm_high_thresh:.3f}, Low: {abs_norm_low_thresh:.3f}")
        logger.info(f"  Slope Decline: {slope_decline_thresh:.4f}, Increase: {slope_increase_thresh:.4f}")
        
        for i in range(height):
            for j in range(width):
                # Skip invalid pixels
                if (median_yield[i, j] == -999 or std_deviation[i, j] == -999):
                    zones[i, j] = 0
                    continue
                
                med_val = median_yield[i, j]
                std_val = std_deviation[i, j]
                abs_val = absolute_norm[i, j] if absolute_norm[i, j] != -999 else 0
                slope_val = trend_slope[i, j] if trend_slope[i, j] != -999 else 0
                r2_val = trend_r2[i, j] if trend_r2[i, j] != -999 else 0
                
                # Priority 1: Significant trends (if R² is high enough)
                if r2_val >= r2_significance:
                    if slope_val <= slope_decline_thresh:
                        zones[i, j] = 4  # Declining Trend
                        continue
                    elif slope_val >= slope_increase_thresh:
                        zones[i, j] = 5  # Increasing Trend
                        continue
                
                # Priority 2: High variability
                if std_val >= std_high_thresh:
                    zones[i, j] = 3  # Variable
                    continue
                
                # Priority 3: Stable zones based on median and absolute normalization
                if med_val >= median_high_thresh and abs_val >= abs_norm_high_thresh:
                    zones[i, j] = 1  # Stable High
                elif med_val <= median_low_thresh and abs_val <= abs_norm_low_thresh:
                    zones[i, j] = 2  # Stable Low
                elif med_val >= median_high_thresh:
                    zones[i, j] = 1  # Stable High (high median, regardless of abs norm)
                elif med_val <= median_low_thresh:
                    zones[i, j] = 2  # Stable Low (low median, regardless of abs norm)
                else:
                    # Middle-range pixels - classify based on absolute normalization
                    if abs_val >= 0:  # Above average relative performance
                        zones[i, j] = 1  # Stable High
                    else:
                        zones[i, j] = 2  # Stable Low
        
        # Log zone statistics
        zone_counts = np.bincount(zones.flatten(), minlength=6)
        total_pixels = np.sum(zone_counts[1:])  # Exclude unclassified
        
        logger.info("Zone Classification Results:")
        logger.info(f"  Unclassified: {zone_counts[0]} pixels")
        logger.info(f"  Stable High: {zone_counts[1]} pixels ({zone_counts[1]/total_pixels*100:.1f}%)")
        logger.info(f"  Stable Low: {zone_counts[2]} pixels ({zone_counts[2]/total_pixels*100:.1f}%)")
        logger.info(f"  Variable: {zone_counts[3]} pixels ({zone_counts[3]/total_pixels*100:.1f}%)")
        logger.info(f"  Declining: {zone_counts[4]} pixels ({zone_counts[4]/total_pixels*100:.1f}%)")
        logger.info(f"  Increasing: {zone_counts[5]} pixels ({zone_counts[5]/total_pixels*100:.1f}%)")
        
        return zones
    
    def save_result(self, data: np.ndarray, filename: str, profile: Dict, 
                   description: str, nodata_value: float = -999):
        """
        Save result as TIFF file.
        
        Args:
            data (np.ndarray): 2D array to save
            filename (str): Output filename
            profile (Dict): Rasterio profile
            description (str): Description for logging
            nodata_value (float): Value to use for nodata pixels
        """
        output_path = self.output_dir / filename
        
        # Update profile for single band output
        output_profile = profile.copy()
        output_profile.update({
            'dtype': 'float32',
            'count': 1,
            'nodata': nodata_value,
            'compress': 'lzw'
        })
        
        try:
            with rasterio.open(output_path, 'w', **output_profile) as dst:
                dst.write(data, 1)
                dst.set_band_description(1, description)
            
            logger.info(f"Saved {description}: {output_path}")
            
            # Print basic statistics
            valid_data = data[data != nodata_value]
            if len(valid_data) > 0:
                logger.info(f"  Min: {np.min(valid_data):.4f}")
                logger.info(f"  Max: {np.max(valid_data):.4f}")
                logger.info(f"  Mean: {np.mean(valid_data):.4f}")
                logger.info(f"  Std: {np.std(valid_data):.4f}")
                logger.info(f"  Valid pixels: {len(valid_data)}")
            
        except Exception as e:
            logger.error(f"Error saving {output_path}: {e}")
    
    def run_analysis(self):
        """Run the complete temporal analysis."""
        logger.info("Starting NDVI temporal analysis...")
        
        if len(self.ndvi_files) < 2:
            logger.error("Need at least 2 years of data for temporal analysis")
            return
        
        # Load all NDVI data
        ndvi_stack, profile, years = self._load_ndvi_data()
        
        logger.info(f"Processing data for years: {', '.join(years)}")
        logger.info(f"Data shape: {ndvi_stack.shape}")
        
        # Calculate all statistics
        median_yield = self.calculate_median_yield_ranking(ndvi_stack)
        variability_ranking = self.calculate_variability_ranking(ndvi_stack)
        std_deviation = self.calculate_standard_deviation(ndvi_stack)
        absolute_norm = self.calculate_absolute_normalization(ndvi_stack)
        trend_slope, trend_r2 = self.calculate_temporal_trend(ndvi_stack)
        
        # Classify management zones
        management_zones = self.classify_management_zones(
            median_yield, std_deviation, absolute_norm, trend_slope, trend_r2
        )
        
        # Save all results
        year_range = f"{years[0]}_{years[-1]}"
        
        # Original statistics
        self.save_result(
            median_yield, 
            f"ndvi_median_yield_ranking_{year_range}.tif",
            profile,
            f"NDVI Median Yield Ranking ({year_range})"
        )
        
        self.save_result(
            variability_ranking,
            f"ndvi_variability_ranking_{year_range}.tif", 
            profile,
            f"NDVI Variability Ranking (95th-5th percentile, {year_range})"
        )
        
        self.save_result(
            std_deviation,
            f"ndvi_standard_deviation_{year_range}.tif",
            profile, 
            f"NDVI Standard Deviation ({year_range})"
        )
        
        # New advanced statistics
        self.save_result(
            absolute_norm,
            f"ndvi_absolute_normalization_{year_range}.tif",
            profile,
            f"NDVI Absolute Normalization ({year_range})"
        )
        
        self.save_result(
            trend_slope,
            f"ndvi_temporal_trend_slope_{year_range}.tif",
            profile,
            f"NDVI Temporal Trend Slope ({year_range})"
        )
        
        self.save_result(
            trend_r2,
            f"ndvi_temporal_trend_r2_{year_range}.tif",
            profile,
            f"NDVI Temporal Trend R-squared ({year_range})"
        )
        
        # Management zones (special handling for categorical data)
        self.save_result(
            management_zones.astype(np.float32),
            f"ndvi_management_zones_{year_range}.tif",
            profile,
            f"NDVI Management Zones ({year_range})",
            nodata_value=0
        )
        
        logger.info("Analysis completed successfully!")
        
        # Create summary report
        self._create_summary_report(years, ndvi_stack.shape)
    
    def _create_summary_report(self, years: List[str], data_shape: Tuple):
        """Create a summary report of the analysis."""
        report_path = self.output_dir / "temporal_analysis_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("NDVI Temporal Analysis Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Analysis Date: {logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', (), None))}\n")
            f.write(f"Input Directory: {self.input_dir}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            f.write(f"Years Analyzed: {', '.join(years)} ({len(years)} years)\n")
            f.write(f"Spatial Dimensions: {data_shape[0]} rows x {data_shape[1]} columns\n\n")
            
            f.write("Generated Files:\n")
            f.write("Basic Statistics:\n")
            f.write("1. ndvi_median_yield_ranking_*.tif - Median NDVI values for yield ranking\n")
            f.write("2. ndvi_variability_ranking_*.tif - Variability (95th - 5th percentile)\n")
            f.write("3. ndvi_standard_deviation_*.tif - Standard deviation across years\n\n")
            
            f.write("Advanced Statistics:\n")
            f.write("4. ndvi_absolute_normalization_*.tif - Absolute normalization across all years\n")
            f.write("5. ndvi_temporal_trend_slope_*.tif - Temporal trend slope (change per year)\n")
            f.write("6. ndvi_temporal_trend_r2_*.tif - R-squared values for temporal trends\n")
            f.write("7. ndvi_management_zones_*.tif - Management zone classifications\n\n")
            
            f.write("Management Zone Classifications:\n")
            f.write("0 = Invalid/NoData (pixels with insufficient data)\n")
            f.write("1 = Stable High (high median performance)\n")
            f.write("2 = Stable Low (low median performance)\n")
            f.write("3 = Variable (high standard deviation)\n")
            f.write("4 = Declining Trend (significant negative slope)\n")
            f.write("5 = Increasing Trend (significant positive slope)\n\n")
            
            f.write("Management Recommendations by Zone:\n")
            f.write("- Stable High: Management is working well, maintain practices\n")
            f.write("- Stable Low: Address soil fertility, drainage, or other limiting factors\n")
            f.write("- Variable: Consider variable rate applications or field experiments\n")
            f.write("- Declining: Investigate consistent problems, may need intervention\n")
            f.write("- Increasing: Identify successful practices for other field areas\n\n")
            
            f.write("Notes:\n")
            f.write("- NoData value: -999 (0 for management zones)\n")
            f.write("- NDVI values should be in normalized range\n")
            f.write("- Absolute normalization shows relative performance across entire field\n")
            f.write("- Trend analysis requires minimum 3 years of data per pixel\n")
            f.write("- Zone classification uses percentile-based thresholds\n")
        
        logger.info(f"Summary report saved: {report_path}")


def main():
    """Main function to run the temporal analysis."""
    parser = argparse.ArgumentParser(description='Calculate NDVI temporal statistics')
    parser.add_argument('--input-dir', '-i', 
                       default='ndvi_outputs',
                       help='Input directory containing NDVI TIFF files')
    parser.add_argument('--output-dir', '-o',
                       default='ndvi_temporal_statistics',
                       help='Output directory for statistics files')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    script_dir = Path(__file__).parent
    input_dir = script_dir / args.input_dir
    output_dir = script_dir / args.output_dir
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Run analysis
    analyzer = NDVITemporalAnalyzer(str(input_dir), str(output_dir))
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
